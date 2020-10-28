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

  return combinedModule;
}

} // namespace spirv
} // namespace mlir
