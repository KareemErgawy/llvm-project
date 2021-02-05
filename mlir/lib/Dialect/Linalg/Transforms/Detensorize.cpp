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
#include <memory>

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Defines the criteria a TensorType must follow in order to be considered
/// "detensorable".
///
/// NOTE: For now, only 0-D are supported.
///
/// Returns true if tensorType can be detensored.
bool canBeDetensored(TensorType tensorType) {
  return tensorType.hasRank() && tensorType.getRank() == 0;
}

/// A conversion patttern for detensoring `linalg.generic` ops.
class DetensorizeGenericOp : public OpConversionPattern<GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GenericOp genericOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation &genericOpBody = genericOp.getBody()->front();
    BlockAndValueMapping tensorToDetensoredOperandMapping;

    tensorToDetensoredOperandMapping.map(
        genericOpBody.getOperands(),
        ArrayRef<Value>{operands.begin(), genericOpBody.getNumOperands()});

    OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPoint(genericOp);
    Operation *detensoredOp =
        genericOpBody.clone(tensorToDetensoredOperandMapping);
    rewriter.insert(detensoredOp);
    rewriter.replaceOp(genericOp, detensoredOp->getResults());

    return success();
  }
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

    // A detensored value is converted back by creating a new tensor from its
    // element(s).
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      auto createNewTensorOp = builder.create<tensor::FromElementsOp>(
          loc, inputs[0].getType(), inputs[0]);

      // FromElementsOp results in a tensor<1xdtype>, we need to reshape that to
      // a tensor<dtype> instead.
      return builder.create<linalg::TensorReshapeOp>(
          loc, type, createNewTensorOp, ArrayRef<ReassociationExprs>{});
    });
  }
};

// Canonicalizes the pattern of the form
//
// %tensor = tensor.from_elements(%element) : (i32) -> tensor<1xi32>
// %reshaped_tensor = linalg.tensor_reshape %tensor [] : tensor<1xi32> into
//   tensor<i32>
// %extracted_element = tensor.extract %reshaped_tensor[] : tensor<i32>
//
// to just %element.
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
  void runOnFunction() override {
    auto *context = &getContext();
    DetensorizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    patterns.insert<DetensorizeGenericOp>(typeConverter, context);

    target.addDynamicallyLegalOp<GenericOp>([&](GenericOp op) {
      // If any of the operands or results cannot be detensored, the op is
      // considered legal and won't be detensored.
      return llvm::any_of(
          op.getShapedOperandTypes(), [](ShapedType shapedType) {
            assert(shapedType.isa<TensorType>());
            return !canBeDetensored(shapedType.cast<TensorType>());
          });
    });

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();

    // a canonicalization pattern to get rid of such op sequences.
    OwningRewritePatternList canonPatterns;
    canonPatterns.insert<ExtractFromReshapeFromElements>(context);

    if (failed(applyPatternsAndFoldGreedily(getFunction(),
                                            std::move(canonPatterns))))
      signalPassFailure();

    // TODO Properly handle control flow within function boundaries.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLinalgDetensorizePass() {
  return std::make_unique<LinalgDetensorize>();
}
