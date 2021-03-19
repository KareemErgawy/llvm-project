//===- Detensorize.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <iterator>
#include <memory>

using namespace mlir;
using namespace mlir::linalg;

static Value sourceMaterializationCallback(OpBuilder &builder, Type type,
                                           ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  // A detensored value is converted back by creating a new tensor from its
  // element(s).
  auto createNewTensorOp = builder.create<tensor::FromElementsOp>(
      loc, inputs[0].getType(), inputs[0]);

  // FromElementsOp results in a tensor<1xdtype>, we need to reshape that to
  // a tensor<dtype> instead.
  return builder.create<linalg::TensorReshapeOp>(
      loc, type, createNewTensorOp, ArrayRef<ReassociationExprs>{});
}

namespace {
/// Defines the criteria a TensorType must follow in order to be considered
/// "detensorable".
///
/// NOTE: For now, only 0-D tensors are supported.
///
/// Returns true if tensorType can be detensored.
bool canBeDetensored(TensorType tensorType) {
  return tensorType.hasRank() && tensorType.getRank() == 0;
}

bool shouldBeDetensored(Operation *op, TypeConverter typeConverter) {
  GenericOp genericOp = dyn_cast_or_null<GenericOp>(op);
  return genericOp && llvm::all_of(genericOp.getShapedOperandTypes(),
                                   [&](ShapedType shapedType) {
                                     return !typeConverter.isLegal(shapedType);
                                   });
}

/// A conversion patttern for detensoring `linalg.generic` ops.
class DetensorizeGenericOp : public OpConversionPattern<GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Block *originalBlock = op->getBlock();

    // Gather some information about the op before inling its region.
    Block *opEntryBlock = &*op.region().begin();
    YieldOp yieldOp = dyn_cast<YieldOp>(op.region().back().getTerminator());

    // Split the op's region before the op. This way, we have a clear insertion
    // point in which the op can be inlined.
    Block *newBlock = originalBlock->splitBlock(op);
    rewriter.inlineRegionBefore(op.region(), newBlock);
    // Now that op's region is inlined, the operands of its YieldOp are mapped
    // to the materialized target values. Therefore, we can replace the op's
    // uses with those of its YielOp's operands.
    rewriter.replaceOp(op, yieldOp->getOperands());

    // No need for these intermediate blocks, merge them into 1.
    rewriter.mergeBlocks(opEntryBlock, originalBlock, operands);
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(&*Block::iterator(yieldOp));

    return success();
  }
};

/// A conversion pattern for detensoring internal (non-entry) blocks within a
/// function.
struct FunctionNonEntryBlockConversion : public ConversionPattern {
  FunctionNonEntryBlockConversion(
      StringRef functionLikeOpName, MLIRContext *ctx, TypeConverter &converter,
      DenseMap<Block *, DenseSet<int>> blockArgumentDetensoring)
      : ConversionPattern(converter, functionLikeOpName, /*benefit=*/1, ctx),
        blockArgumentDetensoring(blockArgumentDetensoring) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    Region &region = mlir::impl::getFunctionBody(op);
    SmallVector<TypeConverter::SignatureConversion, 2> conversions;

    for (Block &block : llvm::drop_begin(region, 1)) {
      conversions.emplace_back(block.getNumArguments());
      TypeConverter::SignatureConversion &back = conversions.back();
      DenseSet<int> blockArgumentDetensoringFilter =
          blockArgumentDetensoring.lookup(&block);

      for (unsigned int idx = 0; idx < block.getNumArguments(); ++idx) {
        if (blockArgumentDetensoringFilter.count(idx))
          back.addInputs(idx, {getTypeConverter()->convertType(
                                  block.getArgumentTypes()[idx])});
        else
          back.addInputs(idx, {block.getArgumentTypes()[idx]});
      }
    }

    if (failed(rewriter.convertNonEntryRegionTypes(&region, *typeConverter,
                                                   &conversions))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  const DenseMap<Block *, DenseSet<int>> blockArgumentDetensoring;
};

class DetensorizeTypeConverter : public TypeConverter {
public:
  DetensorizeTypeConverter() {
    addConversion([](Type type) { return type; });

    // A TensorType that can be detensored, is converted to the underlying
    // element type.
    addConversion([](TensorType tensorType) -> Type {
      if (canBeDetensored(tensorType))
        return tensorType.getElementType();

      return tensorType;
    });

    // A tensor value is detensoried by extracting its element(s).
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<tensor::ExtractOp>(loc, inputs[0], ValueRange{});
    });

    addSourceMaterialization(sourceMaterializationCallback);
    addArgumentMaterialization(sourceMaterializationCallback);
  }
};

/// Canonicalizes the pattern of the form
///
/// %tensor = tensor.from_elements(%element) : (i32) -> tensor<1xi32>
/// %reshaped_tensor = linalg.tensor_reshape %tensor [] : tensor<1xi32> into
///   tensor<i32>
/// %extracted_element = tensor.extract %reshaped_tensor[] : tensor<i32>
///
/// to just %element.
struct ExtractFromReshapeFromElements
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter &rewriter) const final {
    if (extract.indices().size() != 0)
      return failure();

    auto tensorReshape = extract.tensor().getDefiningOp<TensorReshapeOp>();
    if (tensorReshape == nullptr)
      return failure();

    auto tensorFromElements =
        tensorReshape.getOperand()
            .getDefiningOp<mlir::tensor::FromElementsOp>();
    if (tensorFromElements == nullptr)
      return failure();

    rewriter.replaceOp(extract, tensorFromElements.getOperand(0));
    return success();
  }
};

/// @see LinalgDetensorize in Linalg/Passes.td for more details.
struct LinalgDetensorize : public LinalgDetensorizeBase<LinalgDetensorize> {
  LinalgDetensorize() = default;
  LinalgDetensorize(const LinalgDetensorize &pass) {}

  class CostModel {
  public:
    virtual ~CostModel() = default;

    /// A cost model algorithm computes the following outputs:
    ///
    /// - detensorableLinalgOps: the list of linalg ops that should be
    /// detensored.
    ///
    /// - detensorableBranchOps: a map whose keys are branch ops and whose
    /// values are operand indices for such keys. The set of operand indices
    /// corresponding to a branch op specify which sub-set of the branch's
    /// operands should be detensored (i.e. converted by typeConverter).
    ///
    /// - blockArgumentDetensoring: since the operands and results of detensored
    /// lingal ops can cross the BB boundary (e.g. a linalg op's input can come
    /// from a BB argument and a linalg op's output can be passed to successor
    /// BBs), we need to maintain the sub-set of arguments that should be
    /// detensored (i.e. converted by typeConverter) for each affected BB.
    ///
    /// Example:
    ///
    /// For the following snippet:
    /// ...
    /// ^bb1(%6: tensor<i32>, %9: tensor<i32>):
    ///   %7 = linalg.init_tensor [] : tensor<i32>
    ///   %8 = linalg.generic #attrs
    ///     ins(%6, %6 : tensor<i32>, tensor<i32>)
    ///     outs(%7 : tensor<i32>) {
    ///     ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    ///       %9 = addi %arg0, %arg1 : i32
    ///       linalg.yield %9 : i32
    ///   } -> tensor<i32>
    ///   %10 = "some.op"(%9)
    ///   br ^bb2(%8 : tensor<i32>)
    /// ...
    ///
    /// if the cost model decides that the linalg.generic op should be
    /// detensored, then:
    /// - detensorableLinalgOps should be = {linalg.generic{add}}.
    /// - detensorableBranchOps should be = {bb2 -> {0}}.
    /// - blockArgumentDetensoring should be = {bb1 -> {0}, bb2 -> {0}}.
    virtual void
    compute(FuncOp func, DetensorizeTypeConverter typeConverter,
            DenseSet<Operation *> &detensorableLinalgOps,
            DenseMap<Operation *, DenseSet<int>> &detensorableBranchOps,
            DenseMap<Block *, DenseSet<int>> &blockArgumentDetensoring) = 0;
  };

  class PureControlFlowDetectionModel : public CostModel {
    void compute(
        FuncOp func, DetensorizeTypeConverter typeConverter,
        DenseSet<Operation *> &detensorableLinalgOps,
        DenseMap<Operation *, DenseSet<int>> &detensorableBranchOps,
        DenseMap<Block *, DenseSet<int>> &blockArgumentDetensoring) override {
      // TODO The following code is implemented with loops in mind. We might
      // need to add support for if conditions later on.

      DenseSet<Operation *> workList;
      // 1. Find which detensorable ops are involved in control-flow (i.e.
      // they produce tensors that are then used in a cond_br's condition).
      func.walk([&](CondBranchOp condBr) {
        auto *chainOp = condBr.condition().getDefiningOp();

        while (chainOp && !dyn_cast<GenericOp>(chainOp)) {
          if (chainOp->getNumOperands() != 1)
            break;

          chainOp = chainOp->getOperand(0).getDefiningOp();
        }

        if (!shouldBeDetensored(chainOp, typeConverter))
          return;

        workList.insert(chainOp);
      });

      // 2. Discover other detensorable ops by walking the def-use chain
      // backwards starting from the detensorable ops currently on the
      // workList.
      while (!workList.empty()) {
        GenericOp detensorableOp = cast<GenericOp>(*workList.begin());
        detensorableLinalgOps.insert(detensorableOp);
        workList.erase(workList.begin());

        // Discover where the detensorableOp's operands come from.
        for (Value operand : detensorableOp.inputs())
          if (!discoverDetensorableComponent(
                  operand, typeConverter, workList, detensorableLinalgOps,
                  detensorableBranchOps, blockArgumentDetensoring)) {
            // TODO For now we assume there is one opportunity for detensoring
            // in a function. This can be extended to support multiple separate
            // components in a single function.
            detensorableLinalgOps.clear();
            detensorableBranchOps.clear();
            blockArgumentDetensoring.clear();
            return;
          }
      }
    }

  private:
    bool discoverDetensorableComponent(
        Value operand, TypeConverter typeConverter,
        DenseSet<Operation *> &workList,
        const DenseSet<Operation *> &detensorableLinalgOps,
        DenseMap<Operation *, DenseSet<int>> &detensorableBranchOps,
        DenseMap<Block *, DenseSet<int>> &blockArgumentDetensoring) {
      auto *definingOp = operand.getDefiningOp();

      if (definingOp) {
        if (comesFromElements(definingOp))
          return true;

        if (!shouldBeDetensored(definingOp, typeConverter))
          return false;

        if (!workList.count(definingOp) &&
            !detensorableLinalgOps.count(definingOp))
          workList.insert(definingOp);

        return true;
      }

      BlockArgument blockArgument = operand.cast<BlockArgument>();
      Block *ownerBlock = blockArgument.getOwner();

      if (&*ownerBlock->getParent()->begin() == ownerBlock)
        return true;

      blockArgumentDetensoring[ownerBlock].insert(blockArgument.getArgNumber());

      for (PredecessorIterator pred = ownerBlock->pred_begin();
           pred != ownerBlock->pred_end(); ++pred) {
        BranchOpInterface terminator =
            dyn_cast<BranchOpInterface>((*pred)->getTerminator());
        auto ownerBlockOperands =
            terminator.getSuccessorOperands(pred.getSuccessorIndex());

        // TODO Add a test where the same operand is passed more than once to
        // the same block.
        if (!ownerBlockOperands || ownerBlockOperands->empty())
          continue;

        auto operand =
            ownerBlockOperands.getValue()[blockArgument.getArgNumber()];

        for (int idx = ownerBlockOperands->getBeginOperandIndex(),
                 eidx = idx + ownerBlockOperands->size();
             idx < eidx; ++idx)
          if (terminator->getOperand(idx) == operand)
            detensorableBranchOps[terminator].insert(idx);

        if (!discoverDetensorableComponent(
                operand, typeConverter, workList, detensorableLinalgOps,
                detensorableBranchOps, blockArgumentDetensoring)) {

          return false;
        }
      }

      return true;
    }

    bool comesFromElements(Operation *op) {
      while (op && !dyn_cast<tensor::FromElementsOp>(op)) {
        if (op->getNumOperands() > 1)
          return false;

        op = op->getOperand(0).getDefiningOp();
      }

      return op;
    }
  };

  /// Detensorize everything that can detensored.
  class AggressiveDetensoringModel : public CostModel {
    void compute(
        FuncOp func, DetensorizeTypeConverter typeConverter,
        DenseSet<Operation *> &detensorableLinalgOps,
        DenseMap<Operation *, DenseSet<int>> &detensorableBranchOps,
        DenseMap<Block *, DenseSet<int>> &blockArgumentDetensoring) override {
      func.walk([&](GenericOp genericOp) {
        if (shouldBeDetensored(genericOp, typeConverter))
          detensorableLinalgOps.insert(genericOp);
      });

      func.walk([&](BranchOpInterface brOp) {
        DenseSet<int> brOpOperandDetensoring;

        for (int p = 0, e = brOp->getBlock()->getNumSuccessors(); p < e; ++p) {
          auto successorOperands = brOp.getSuccessorOperands(p);
          Block *successor = brOp->getSuccessor(p);

          if (!successorOperands.hasValue())
            break;

          for (int idx = successorOperands->getBeginOperandIndex(),
                   eidx = idx + successorOperands->size();
               idx < eidx; ++idx) {
            brOpOperandDetensoring.insert(idx);
            blockArgumentDetensoring[successor].insert(
                idx - successorOperands->getBeginOperandIndex());
          }
        }

        detensorableBranchOps.try_emplace(brOp,
                                          std::move(brOpOperandDetensoring));
      });
    }
  };

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    DetensorizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    DenseSet<Operation *> detensorableLinalgOps;
    DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;
    DenseMap<Block *, DenseSet<int>> blockArgumentDetensoring;

    std::unique_ptr<CostModel> costModel;

    if (aggressiveMode.getValue())
      costModel = std::make_unique<AggressiveDetensoringModel>();
    else
      costModel = std::make_unique<PureControlFlowDetectionModel>();

    costModel->compute(getFunction(), typeConverter, detensorableLinalgOps,
                       detensorableBranchOps, blockArgumentDetensoring);

    target.addDynamicallyLegalOp<GenericOp>(
        [&](GenericOp op) { return !detensorableLinalgOps.count(op); });

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      // A function is legal if all of its non-entry blocks are legal. We
      // don't legalize the entry block (i.e. the function's signature) since
      // detensoring can't happen along external calling convention
      // boundaries, which we conservatively approximate as all function
      // signatures.
      return llvm::all_of(llvm::drop_begin(op.getBody(), 1), [&](Block &block) {
        if (blockArgumentDetensoring.count(&block) &&
            llvm::any_of(blockArgumentDetensoring[&block], [&](int idx) {
              return !typeConverter.isLegal(block.getArgumentTypes()[idx]);
            })) {
          return false;
        }
        return true;
      });
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (isNotBranchOpInterfaceOrReturnLikeOp(op) ||
          isLegalForReturnOpTypeConversionPattern(op, typeConverter,
                                                  /*returnOpAlwaysLegal*/ true))
        return true;

      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        if (!detensorableBranchOps.count(branchOp))
          return true;

        for (auto operandIdx : detensorableBranchOps[branchOp])
          if (!typeConverter.isLegal(
                  branchOp->getOperand(operandIdx).getType()))
            return false;

        return true;
      }

      return false;
    });

    patterns.insert<DetensorizeGenericOp>(typeConverter, context);
    patterns.insert<FunctionNonEntryBlockConversion>(FuncOp::getOperationName(),
                                                     context, typeConverter,
                                                     blockArgumentDetensoring);
    // Since non-entry block arguments get detensorized, we also need to
    // update the control flow inside the function to reflect the correct
    // types.
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter,
                                                   &detensorableBranchOps);

    if (failed(applyFullConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet canonPatterns(context);
    canonPatterns.add<ExtractFromReshapeFromElements>(context);
    if (failed(applyPatternsAndFoldGreedily(getFunction(),
                                            std::move(canonPatterns))))
      signalPassFailure();
  }

  Option<bool> aggressiveMode{
      *this, "aggressive-mode",
      llvm::cl::desc("Detensorize all ops that qualify for detensoring along "
                     "with branch operands and basic-block arguments.")};
};
} // namespace

std::unique_ptr<Pass> mlir::createLinalgDetensorizePass() {
  return std::make_unique<LinalgDetensorize>();
}
