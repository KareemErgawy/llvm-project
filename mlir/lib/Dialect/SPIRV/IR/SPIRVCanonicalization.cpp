//===- SPIRVCanonicalization.cpp - MLIR SPIR-V canonicalization patterns --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the folders and canonicalization patterns for SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

/// Returns the boolean value under the hood if the given `boolAttr` is a scalar
/// or splat vector bool constant.
static Optional<bool> getScalarOrSplatBoolAttr(Attribute boolAttr) {
  if (!boolAttr)
    return llvm::None;

  auto type = boolAttr.getType();
  if (type.isInteger(1)) {
    auto attr = boolAttr.cast<BoolAttr>();
    return attr.getValue();
  }
  if (auto vecType = type.cast<VectorType>()) {
    if (vecType.getElementType().isInteger(1))
      if (auto attr = boolAttr.dyn_cast<SplatElementsAttr>())
        return attr.getSplatValue<bool>();
  }
  return llvm::None;
}

// Extracts an element from the given `composite` by following the given
// `indices`. Returns a null Attribute if error happens.
static Attribute extractCompositeElement(Attribute composite,
                                         ArrayRef<unsigned> indices) {
  // Check that given composite is a constant.
  if (!composite)
    return {};
  // Return composite itself if we reach the end of the index chain.
  if (indices.empty())
    return composite;

  if (auto vector = composite.dyn_cast<ElementsAttr>()) {
    assert(indices.size() == 1 && "must have exactly one index for a vector");
    return vector.getValue({indices[0]});
  }

  if (auto array = composite.dyn_cast<ArrayAttr>()) {
    assert(!indices.empty() && "must have at least one index for an array");
    return extractCompositeElement(array.getValue()[indices[0]],
                                   indices.drop_front());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'erated canonicalizers
//===----------------------------------------------------------------------===//

namespace {
#include "SPIRVCanonicalization.inc"
}

//===----------------------------------------------------------------------===//
// spv.AccessChainOp
//===----------------------------------------------------------------------===//

namespace {

/// Combines chained `spirv::AccessChainOp` operations into one
/// `spirv::AccessChainOp` operation.
struct CombineChainedAccessChain
    : public OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern<spirv::AccessChainOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::AccessChainOp accessChainOp,
                                PatternRewriter &rewriter) const override {
    auto parentAccessChainOp = dyn_cast_or_null<spirv::AccessChainOp>(
        accessChainOp.base_ptr().getDefiningOp());

    if (!parentAccessChainOp) {
      return failure();
    }

    // Combine indices.
    SmallVector<Value, 4> indices(parentAccessChainOp.indices());
    indices.append(accessChainOp.indices().begin(),
                   accessChainOp.indices().end());

    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
        accessChainOp, parentAccessChainOp.base_ptr(), indices);

    return success();
  }
};
} // end anonymous namespace

void spirv::AccessChainOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<CombineChainedAccessChain>(context);
}

//===----------------------------------------------------------------------===//
// spv.BitcastOp
//===----------------------------------------------------------------------===//

void spirv::BitcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<ConvertChainedBitcast>(context);
}

//===----------------------------------------------------------------------===//
// spv.CompositeExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::CompositeExtractOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "spv.CompositeExtract expects one operand");
  auto indexVector =
      llvm::to_vector<8>(llvm::map_range(indices(), [](Attribute attr) {
        return static_cast<unsigned>(attr.cast<IntegerAttr>().getInt());
      }));
  return extractCompositeElement(operands[0], indexVector);
}

//===----------------------------------------------------------------------===//
// spv.Constant
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "spv.Constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IAddOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.IAdd expects two operands");
  // x + 0 = x
  if (matchPattern(operand2(), m_Zero()))
    return operand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IMulOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.IMul expects two operands");
  // x * 0 == 0
  if (matchPattern(operand2(), m_Zero()))
    return operand2();
  // x * 1 = x
  if (matchPattern(operand2(), m_One()))
    return operand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ISubOp::fold(ArrayRef<Attribute> operands) {
  // x - x = 0
  if (operand1() == operand2())
    return Builder(getContext()).getIntegerAttr(getType(), 0);

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// spv.LogicalAnd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalAndOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.LogicalAnd should take two operands");

  if (Optional<bool> rhs = getScalarOrSplatBoolAttr(operands.back())) {
    // x && true = x
    if (rhs.getValue())
      return operand1();

    // x && false = false
    if (!rhs.getValue())
      return operands.back();
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

void spirv::LogicalNotOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .add<ConvertLogicalNotOfIEqual, ConvertLogicalNotOfINotEqual,
           ConvertLogicalNotOfLogicalEqual, ConvertLogicalNotOfLogicalNotEqual>(
          context);
}

//===----------------------------------------------------------------------===//
// spv.LogicalOr
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalOrOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.LogicalOr should take two operands");

  if (auto rhs = getScalarOrSplatBoolAttr(operands.back())) {
    if (rhs.getValue())
      // x || true = true
      return operands.back();

    // x || false = x
    if (!rhs.getValue())
      return operand1();
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spv.mlir.selection
//===----------------------------------------------------------------------===//

namespace {
// Blocks from the given `spv.mlir.selection` operation must satisfy the
// following layout:
//
//       +-----------------------------------------------+
//       | header block                                  |
//       | spv.BranchConditionalOp %cond, ^case0, ^case1 |
//       +-----------------------------------------------+
//                            /   \
//                             ...
//
//
//   +------------------------+    +------------------------+
//   | case #0                |    | case #1                |
//   | spv.Store %ptr %value0 |    | spv.Store %ptr %value1 |
//   | spv.Branch ^merge      |    | spv.Branch ^merge      |
//   +------------------------+    +------------------------+
//
//
//                             ...
//                            \   /
//                              v
//                       +-------------+
//                       | merge block |
//                       +-------------+
//
struct ConvertSelectionOpToSelect
    : public OpRewritePattern<spirv::SelectionOp> {
  using OpRewritePattern<spirv::SelectionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::SelectionOp selectionOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "^^^^ Start SelectionOp canonicalization.\n";
    auto *op = selectionOp.getOperation();
    auto &body = op->getRegion(0);
    // Verifier allows an empty region for `spv.mlir.selection`.
    if (body.empty()) {
      return failure();
    }

    // Check that region consists of 4 blocks:
    // header block, `true` block, `false` block and merge block.
    if (std::distance(body.begin(), body.end()) != 4) {
      return failure();
    }

    auto *headerBlock = selectionOp.getHeaderBlock();
    if (!onlyContainsBranchConditionalOp(headerBlock)) {
      return failure();
    }

    auto brConditionalOp =
        cast<spirv::BranchConditionalOp>(headerBlock->front());

    auto *trueBlock = brConditionalOp.getSuccessor(0);
    auto *falseBlock = brConditionalOp.getSuccessor(1);
    auto *mergeBlock = selectionOp.getMergeBlock();

    if (failed(canCanonicalizeSelection(trueBlock, falseBlock, mergeBlock)))
      return failure();

    auto trueValue = getSrcValue(trueBlock);
    auto falseValue = getSrcValue(falseBlock);
    auto ptrValue = getDstPtr(trueBlock);
    auto storeOpAttributes =
        cast<spirv::StoreOp>(trueBlock->front())->getAttrs();

    auto selectOp = rewriter.create<spirv::SelectOp>(
        selectionOp.getLoc(), trueValue.getType(), brConditionalOp.condition(),
        trueValue, falseValue);
    rewriter.create<spirv::StoreOp>(selectOp.getLoc(), ptrValue,
                                    selectOp.getResult(), storeOpAttributes);

    // `spv.mlir.selection` is not needed anymore.
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks that given blocks follow the following rules:
  // 1. Each conditional block consists of two operations, the first operation
  //    is a `spv.Store` and the last operation is a `spv.Branch`.
  // 2. Each `spv.Store` uses the same pointer and the same memory attributes.
  // 3. A control flow goes into the given merge block from the given
  //    conditional blocks.
  LogicalResult canCanonicalizeSelection(Block *trueBlock, Block *falseBlock,
                                         Block *mergeBlock) const;

  bool onlyContainsBranchConditionalOp(Block *block) const {
    return std::next(block->begin()) == block->end() &&
           isa<spirv::BranchConditionalOp>(block->front());
  }

  bool isSameAttrList(spirv::StoreOp lhs, spirv::StoreOp rhs) const {
    return lhs->getAttrDictionary() == rhs->getAttrDictionary();
  }

  // Returns a source value for the given block.
  Value getSrcValue(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.value();
  }

  // Returns a destination value for the given block.
  Value getDstPtr(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.ptr();
  }
};

LogicalResult ConvertSelectionOpToSelect::canCanonicalizeSelection(
    Block *trueBlock, Block *falseBlock, Block *mergeBlock) const {
  // Each block must consists of 2 operations.
  if ((std::distance(trueBlock->begin(), trueBlock->end()) != 2) ||
      (std::distance(falseBlock->begin(), falseBlock->end()) != 2)) {
    return failure();
  }

  auto trueBrStoreOp = dyn_cast<spirv::StoreOp>(trueBlock->front());
  auto trueBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(trueBlock->begin()));
  auto falseBrStoreOp = dyn_cast<spirv::StoreOp>(falseBlock->front());
  auto falseBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(falseBlock->begin()));

  if (!trueBrStoreOp || !trueBrBranchOp || !falseBrStoreOp ||
      !falseBrBranchOp) {
    return failure();
  }

  // Checks that given type is valid for `spv.SelectOp`.
  // According to SPIR-V spec:
  // "Before version 1.4, Result Type must be a pointer, scalar, or vector.
  // Starting with version 1.4, Result Type can additionally be a composite type
  // other than a vector."
  bool isScalarOrVector = trueBrStoreOp.value()
                              .getType()
                              .cast<spirv::SPIRVType>()
                              .isScalarOrVector();

  // Check that each `spv.Store` uses the same pointer, memory access
  // attributes and a valid type of the value.
  if ((trueBrStoreOp.ptr() != falseBrStoreOp.ptr()) ||
      !isSameAttrList(trueBrStoreOp, falseBrStoreOp) || !isScalarOrVector) {
    return failure();
  }

  if ((trueBrBranchOp->getSuccessor(0) != mergeBlock) ||
      (falseBrBranchOp->getSuccessor(0) != mergeBlock)) {
    return failure();
  }

  return success();
}
} // end anonymous namespace

void spirv::SelectionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<ConvertSelectionOpToSelect>(context);
}

namespace {
struct ConvertLoopOpToStructuredLoop : public OpRewritePattern<spirv::LoopOp> {
  using OpRewritePattern<spirv::LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    Block *loopOpPredBlock = loopOp->getBlock();
    // We want to branch into the new structured loop. For that, split the
    // loop's block into 2 so that we can insert a BranchOp at the end of the
    // new predecessor block.
    Block *loopOpNewBlock = loopOpPredBlock->splitBlock(loopOp);
    // We want to branch out of the new structured loop. For that, split the
    // loop's block further into 2 so that we can branch from the structured
    // loop's merge block to the new successsor block.
    Block *loopOpSuccBlock =
        loopOpNewBlock->splitBlock(std::next(Block::iterator(loopOp)));

    // Collect loop components.
    //
    // entryBlock is the block that will be updated to contain the
    // spv.mlir.structured_branch.
    Block *entryBlock = loopOp.getEntryBlock();
    Block *headerBlock = loopOp.getHeaderBlock();
    Block *continueBlock = loopOp.getContinueBlock();
    Block *mergeBlock = loopOp.getMergeBlock();
    spirv::BranchOp entryToHeaderBranch =
        dyn_cast<spirv::BranchOp>(entryBlock->getTerminator());
    assert(entryToHeaderBranch &&
           "expected entry-to-header branch to be a BranchOp");

    llvm::errs() << ">>>> entry: \n";
    entryBlock->dump();

    llvm::errs() << ">>>> header: \n";
    headerBlock->dump();

    llvm::errs() << ">>>> continue: \n";
    continueBlock->dump();

    llvm::errs() << ">>>> merge: \n";
    mergeBlock->dump();

    // TODO Find the back edge block properly (DFS from the continue block). For
    // now, we assume that the loop's continue construct consists only of a
    // single block.
    Block *backEdgeBlock = loopOp.getContinueBlock();

    // TODO Construct the range of blocks that starts at the loop entry and ends
    // at the loop header (inclusive) properly. For now, we assume this range
    // consists only of 2 blocks: the entry block and the header block.
    SpilledValuesMap spilledValues = collectValuesSpilledToContinueConstruct(
        {entryBlock, headerBlock}, continueBlock);

    Block *newContinueBlock = updateHeaderToContinueBranch(
        loopOp.getLoc(), headerBlock, continueBlock, spilledValues, rewriter);

    rewriter.inlineRegionBefore(loopOp.getRegion(), loopOpNewBlock);

    // Re-wire the back-edge block to entryBlock since entryBlock is the block
    // that will contain the spv.mlir.structure_br after the re-write.
    rewriter.setInsertionPointToEnd(backEdgeBlock);
    rewriter.eraseOp(backEdgeBlock->getTerminator());
    rewriter.create<spirv::BranchOp>(loopOp.getLoc(), entryBlock);

    // Branch from outside the loop to entryBlock.
    rewriter.setInsertionPointToEnd(loopOpPredBlock);
    rewriter.create<spirv::BranchOp>(loopOp.getLoc(), entryBlock);

    // Branch from entryBlock to headerBlock.
    rewriter.setInsertionPointToEnd(entryBlock);
    StringAttr loopCFAttr =
        StringAttr::get(loopOp.getContext(),
                        spirv::stringifyControlFlow(spirv::ControlFlow::Loop));
    rewriter.create<spirv::StructuredBranchOp>(
        loopOp.getLoc(), loopCFAttr, entryToHeaderBranch.getOperands(),
        headerBlock, mergeBlock, newContinueBlock);
    rewriter.eraseOp(entryToHeaderBranch);

    // Branch from mergeBlock to after the loop.
    rewriter.setInsertionPointToEnd(mergeBlock);
    rewriter.eraseOp(mergeBlock->getTerminator());
    rewriter.create<spirv::BranchOp>(loopOp.getLoc(), loopOpSuccBlock);

    // The loopOp's region has been inlined and re-wired using
    // spv.mlir.structured_branch, no need for the loopOp anymore.
    rewriter.eraseOp(loopOp);

    return success();
  }

private:
  using SpilledValuesMap = llvm::DenseMap<Value, llvm::DenseSet<OpOperand *>>;

  /// Discovers all the values defined by any block in headerToEntryRange which
  /// spill into (i.e. are used by) the loop's continue construct. See:
  /// createNewContinueBlock(...) for why this is needed.
  ///
  /// Returns a map from spilled values to their uses in the loop's continue
  /// construct.
  SpilledValuesMap
  collectValuesSpilledToContinueConstruct(mlir::BlockRange headerToEntryRange,
                                          Block *continueBlock) const {
    llvm::DenseSet<Value> headerToEntryDefinedValues;

    for (Block *block : headerToEntryRange) {
      for (BlockArgument &blockArg : block->getArguments()) {
        headerToEntryDefinedValues.insert(blockArg);
      }

      for (Operation &op : block->getOperations()) {
        for (OpResult opRes : op.getResults()) {
          headerToEntryDefinedValues.insert(opRes);
        }
      }
    }

    SpilledValuesMap spilledValues;

    // TODO Instead of this, walk the use list of headerToEntryDefinedValues
    // and check for uses that lie in the continue construct.
    for (Operation &op : continueBlock->getOperations()) {
      for (OpOperand &opOperand : op.getOpOperands()) {
        if (headerToEntryDefinedValues.count(opOperand.get())) {
          spilledValues[opOperand.get()].insert(&opOperand);
        }
      }
    }

    return spilledValues;
  }

  /// Creates a new continue block that will explicitly take as arguments the
  /// set of spilled values discovered by
  /// collectValuesSpilledToContinueConstruct(...).
  Block *createNewContinueBlock(Location loc, Block *loopEntry,
                                Block *continueBlock,
                                SpilledValuesMap spilledValues,
                                PatternRewriter &rewriter) const {
    SmallVector<Type> spilledValueTypes;

    for (auto spilledValue : spilledValues) {
      spilledValueTypes.push_back(spilledValue.first.getType());
    }

    Block *newContinueBlock =
        rewriter.createBlock(continueBlock, spilledValueTypes);
    rewriter.setInsertionPointToEnd(newContinueBlock);
    rewriter.create<spirv::BranchOp>(loc, continueBlock);

    ArrayRef<BlockArgument> newContinueBlockArgs =
        newContinueBlock->getArguments();

    unsigned spilledValueIndex = 0;

    for (auto spilledValue : spilledValues) {
      spilledValue.first.replaceUsesWithIf(
          newContinueBlockArgs[spilledValueIndex], [&](OpOperand &userOperand) {
            return spilledValue.second.count(&userOperand);
          });

      ++spilledValueIndex;
    }

    return newContinueBlock;
  }

  /// Rewrite the branch from the loop's header to its continue construct. The
  /// new branch points to a new block that explicitly takes as argument the
  /// values spilled into the continue construct from the entry-to-header range
  /// of blocks.
  //
  // TODO For now, we assume that there are no intervening blocks between the
  // loop's header and continue blocks.
  Block *updateHeaderToContinueBranch(Location loc, Block *loopEntry,
                                      Block *continueBlock,
                                      SpilledValuesMap spilledValues,
                                      PatternRewriter &rewriter) const {

    Block *newContinueBlock = createNewContinueBlock(
        loc, loopEntry, continueBlock, spilledValues, rewriter);

    Operation *entryTerminator = loopEntry->getTerminator();

    if (auto branchCondOp =
            dyn_cast<spirv::BranchConditionalOp>(entryTerminator)) {
      SmallVector<Value, 4> trueArguments = branchCondOp.trueTargetOperands();
      SmallVector<Value, 4> falseArguments = branchCondOp.falseTargetOperands();
      SmallVectorImpl<Value> *affectedArgumentsList = nullptr;

      Block *trueTarget = branchCondOp.trueTarget();
      Block *falseTarget = branchCondOp.falseTarget();

      if (continueBlock == branchCondOp.trueTarget()) {
        affectedArgumentsList = &trueArguments;
        trueTarget = newContinueBlock;
      } else if (continueBlock == branchCondOp.falseTarget()) {
        affectedArgumentsList = &falseArguments;
        falseTarget = newContinueBlock;
      } else {
        branchCondOp.emitError(
            "expected continue block to be a successor of entry block");
      }

      for (auto spilledValue : spilledValues) {
        affectedArgumentsList->push_back(spilledValue.first);
      }

      rewriter.setInsertionPointToEnd(loopEntry);
      rewriter.eraseOp(entryTerminator);
      rewriter.create<spirv::BranchConditionalOp>(
          branchCondOp.getLoc(), branchCondOp.condition(), trueArguments,
          falseArguments, branchCondOp.branch_weightsAttr(), trueTarget,
          falseTarget);
    } else {
      entryTerminator->emitError(
          "unimplemented terminator for loop conversion");
    }

    return newContinueBlock;
  }
};
} // end anonymous namespace

void spirv::LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<ConvertLoopOpToStructuredLoop>(context);
}
