//===- ModuleCombiner.cpp - MLIR SPIR-V Module Combiner ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the the SPIR-V module combiner library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/ModuleCombiner.h"

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

static SmallString<64> renameSymbol(StringRef oldSymName, unsigned &nextFreeID,
                                    spirv::ModuleOp combinedModule) {
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');
  while (true) {
    std::string possible = (newSymName + llvm::utostr(++nextFreeID)).str();

    if (!SymbolTable::lookupSymbolIn(combinedModule, possible)) {
      newSymName += llvm::utostr(nextFreeID);
      break;
    }
  }

  return newSymName;
}

/// Check if a symbol with the same name as op already exists in source. If so,
/// rename op and update all its references in target.
static LogicalResult updateSymbolAndAllUses(SymbolOpInterface op,
                                            spirv::ModuleOp target,
                                            spirv::ModuleOp source,
                                            unsigned &nextFreeID) {
  if (!SymbolTable::lookupSymbolIn(source, op.getName()))
    return success();

  StringRef oldSymName = op.getName();
  SmallString<64> newSymName = renameSymbol(oldSymName, nextFreeID, target);

  if (failed(SymbolTable::replaceAllSymbolUses(op, newSymName, target)))
    return op.emitError("unable to update all symbol uses for ")
           << oldSymName << " to " << newSymName;

  SymbolTable::setSymbolName(op, newSymName);
  return success();
}

namespace mlir {
namespace spirv {

/// This class contains the information for comparing the equivalencies of two
/// blocks. Blocks are considered equivalent if they contain the same operations
/// in the same order. The only allowed divergence is for operands that come
/// from sources outside of the parent block, i.e. the uses of values produced
/// within the block must be equivalent.
///   e.g.,
/// Equivalent:
///  ^bb1(%arg0: i32)
///    return %arg0, %foo : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
/// Not Equivalent:
///  ^bb1(%arg0: i32)
///    return %foo, %arg0 : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
struct BlockEquivalenceData {
  BlockEquivalenceData(Block *block);

  /// Return the order index for the given value that is within the block of
  /// this data.
  unsigned getOrderOf(Value value) const;

  /// The block this data refers to.
  Block *block;
  /// A hash value for this block.
  llvm::hash_code hash;
  /// A map of result producing operations to their relative orders within this
  /// block. The order of an operation is the number of defined values that are
  /// produced within the block before this operation.
  DenseMap<Operation *, unsigned> opOrderIndex;
};

BlockEquivalenceData::BlockEquivalenceData(Block *block)
    : block(block), hash(0) {
  unsigned orderIt = block->getNumArguments();
  for (Operation &op : *block) {
    if (unsigned numResults = op.getNumResults()) {
      opOrderIndex.try_emplace(&op, orderIt);
      orderIt += numResults;
    }
    auto opHash = OperationEquivalence::computeHash(
        &op, OperationEquivalence::Flags::IgnoreOperands);
    hash = llvm::hash_combine(hash, opHash);
  }
}

unsigned BlockEquivalenceData::getOrderOf(Value value) const {
  assert(value.getParentBlock() == block && "expected value of this block");

  // Arguments use the argument number as the order index.
  if (BlockArgument arg = value.dyn_cast<BlockArgument>())
    return arg.getArgNumber();

  // Otherwise, the result order is offset from the parent op's order.
  OpResult result = value.cast<OpResult>();
  auto opOrderIt = opOrderIndex.find(result.getDefiningOp());
  assert(opOrderIt != opOrderIndex.end() && "expected op to have an order");
  return opOrderIt->second + result.getResultNumber();
}

spirv::ModuleOp combine(llvm::SmallVectorImpl<spirv::ModuleOp> &modules,
                        OpBuilder &combinedModuleBuilder) {
  unsigned nextFreeID = 0;

  if (modules.empty())
    return nullptr;

  auto addressingModel = modules[0].addressing_model();
  auto memoryModel = modules[0].memory_model();

  auto combinedModule = combinedModuleBuilder.create<spirv::ModuleOp>(
      modules[0].getLoc(), addressingModel, memoryModel);
  combinedModuleBuilder.setInsertionPointToStart(&*combinedModule.getBody());

  for (auto module : modules) {
    if (module.addressing_model() != addressingModel ||
        module.memory_model() != memoryModel) {
      module.emitError(
          "input modules differ in addressing model and/or memory model");
      return nullptr;
    }

    spirv::ModuleOp moduleClone = module.clone();

    // In the combined module, rename all symbols that conflict with symbols
    // from the current input module. This renmaing applies to all ops except
    // for spv.funcs. This way, if the conflicting op in the input module is
    // non-spv.func, we rename that symbol instead and maintain the spv.func in
    // the combined module name as it is.
    for (auto &op : combinedModule.getBlock().without_terminator()) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op))
        if (!dyn_cast<FuncOp>(op) &&
            failed(updateSymbolAndAllUses(symbolOp, combinedModule, moduleClone,
                                          nextFreeID))) {
          return nullptr;
        }
    }

    // In the current input module, rename all symbols that conflict with
    // symbols from the combined module. This includes renaming spv.funcs.
    for (auto &op : moduleClone.getBlock().without_terminator()) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op))
        if (failed(updateSymbolAndAllUses(symbolOp, moduleClone, combinedModule,
                                          nextFreeID))) {
          return nullptr;
        }
    }

    // Clone all the module's ops to the combined module.
    for (auto &op : moduleClone.getBlock().without_terminator())
      combinedModuleBuilder.insert(op.clone());
  }

  DenseMap<std::pair<int64_t, int64_t>, spirv::GlobalVariableOp>
      descriptorToGlobalVarOpMap;
  DenseMap<StringRef, spirv::GlobalVariableOp> builtInToGlobalVarOpMap;

  combinedModule.walk([&](spirv::GlobalVariableOp globalVarOp) {
    StringRef replacementSymName;

    IntegerAttr descriptorSet = globalVarOp.getAttrOfType<IntegerAttr>(
        spirv::SPIRVDialect::getAttributeName(
            spirv::Decoration::DescriptorSet));
    IntegerAttr binding = globalVarOp.getAttrOfType<IntegerAttr>(
        spirv::SPIRVDialect::getAttributeName(spirv::Decoration::Binding));

    if (descriptorSet) {
      auto result = descriptorToGlobalVarOpMap.try_emplace(
          {descriptorSet.getInt(), binding.getInt()}, globalVarOp);

      // No global variable with the same (descriptor set, binding) was
      // encountered before.
      if (result.second)
        return WalkResult::advance();

      // TODO What if 2 global variables agree on the descriptor set, binding
      // but differ in type?

      replacementSymName = result.first->second.sym_name();
    }

    StringAttr builtIn = globalVarOp.getAttrOfType<StringAttr>(
        spirv::SPIRVDialect::getAttributeName(spirv::Decoration::BuiltIn));

    if (builtIn) {
      auto result =
          builtInToGlobalVarOpMap.try_emplace(builtIn.getValue(), globalVarOp);

      // No global varialbe with the same built-in attribute was encountered
      // before.
      if (result.second)
        return WalkResult::advance();

      replacementSymName = result.first->second.sym_name();
    }

    if (replacementSymName.empty())
      return WalkResult::advance();

    if (failed(SymbolTable::replaceAllSymbolUses(
            globalVarOp, replacementSymName, combinedModule)))
      return WalkResult(
          globalVarOp.emitError("unable to update all symbol uses for ")
          << globalVarOp.sym_name() << " to " << replacementSymName);

    globalVarOp.erase();
    return WalkResult::advance();
  });

  DenseMap<int64_t, spirv::SpecConstantOp> specIdToSpecConstCompositeMap;

  combinedModule.walk([&](spirv::SpecConstantOp specConstOp) {
    IntegerAttr specId = specConstOp.getAttrOfType<IntegerAttr>(
        spirv::SPIRVDialect::getAttributeName(spirv::Decoration::SpecId));

    if (specId) {
      auto result = specIdToSpecConstCompositeMap.try_emplace(specId.getInt(),
                                                              specConstOp);

      // No spec constant with the same spec ID was encountered before.
      if (result.second)
        return WalkResult::advance();

      // TODO What if 2 spec constants agree on the spec ID but differ in types
      // or default values?

      StringRef replacementSymName = result.first->second.sym_name();
      if (failed(SymbolTable::replaceAllSymbolUses(
              specConstOp, replacementSymName, combinedModule)))
        return WalkResult(
            specConstOp.emitError("unable to update all symbol uses for ")
            << specConstOp.sym_name() << " to " << replacementSymName);

      specConstOp.erase();
    }

    return WalkResult::advance();
  });

  // For funcOps, I think that OperationEquivalence won't be useful in detecting
  // whether 2 functions are equivelance or not. The main reason being that
  // OperationEquivalence used Operation::getMutableAttrDict() to compare 2 ops
  // which eventually boils down to comparison of 2 pointer values of the
  // underlying Attribute implementation object.

  combinedModule.walk([&](FuncOp fun1) {
    llvm::hash_code fun1Code(0);
    llvm::errs() << fun1.getName();
    fun1Code = llvm::hash_combine(fun1Code, fun1.type());

    for (auto &blk1 : fun1) {
      BlockEquivalenceData bed(&blk1);
      llvm::errs() << " ==> [blk # " << bed.hash << "] ";
      fun1Code = llvm::hash_combine(fun1Code, bed.hash);
    }

    llvm::errs() << fun1Code << "\n";
  });

  return combinedModule;
}

} // namespace spirv
} // namespace mlir
