#include "mlir/Dialect/SPIRV/ModuleCombiner.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace spirv;

static SmallString<64> renameSymbol(StringRef oldSymName,
                                    unsigned &nextConflictID,
                                    const spirv::ModuleOp combinedModule) {
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');
  while (true) {
    newSymName += llvm::utostr(++nextConflictID);

    if (!SymbolTable::lookupSymbolIn(combinedModule, newSymName))
      break;
  }

  return newSymName;
}

/// Walks the target module for operations of type OpTy. And for each such
/// operation, checks if another operation in the source module has the same
/// symbol. If this is the case, renames the visited/walked operation and
/// updates its references.
template <typename OpTy>
static void updateSymbolAndAllUses(spirv::ModuleOp target,
                                   spirv::ModuleOp source,
                                   unsigned &nextConflictID) {
  target.walk([&](OpTy globalVarOp) {
    if (SymbolTable::lookupSymbolIn(source, globalVarOp.sym_name())) {
      StringRef oldSymName = globalVarOp.sym_name();
      SmallString<64> newSymName =
          renameSymbol(oldSymName, nextConflictID, target);

      if (failed(SymbolTable::replaceAllSymbolUses(globalVarOp, newSymName,
                                                   target)))
        globalVarOp.emitError("unable to update all symbol uses for ")
            << oldSymName << " to " << newSymName;

      SymbolTable::setSymbolName(globalVarOp, newSymName);
    }
  });
}

namespace mlir {
namespace spirv {

void combine(llvm::SmallVector<ModuleOp, 4> modules,
             OpBuilder &combinedModuleBuilder) {
  unsigned nextConflictID = 0;

  if (modules.empty())
    return;

  auto addressingModel = modules[0].addressing_model();
  auto memoryModel = modules[0].memory_model();

  auto combinedModule = combinedModuleBuilder.create<spirv::ModuleOp>(
      modules[0].getLoc(), addressingModel, memoryModel);
  combinedModuleBuilder.setInsertionPointToStart(&*combinedModule.getBody());

  for (auto module : modules) {
    if (module.addressing_model() != addressingModel ||
        module.memory_model() != memoryModel) {
      combinedModule.emitError(
          "input modules differ in addressing model and/or memory model");
      return;
    }

    spirv::ModuleOp moduleClone = module.clone();

    // A global variable is renamed if it conflicts with a function or a spec
    // constant:
    //   (1) Rename global variables from the current input module that are
    //   conflicting with any other module-level op currently in the combined
    //   module.
    updateSymbolAndAllUses<GlobalVariableOp>(moduleClone, combinedModule,
                                             nextConflictID);
    //   (2) Rename global variables currently in the combined module that are
    //   conflicting with any other module-level op in the current input module.
    updateSymbolAndAllUses<GlobalVariableOp>(combinedModule, moduleClone,
                                             nextConflictID);

    // A spec constant is renamed if it conflicts with a function:
    //   (1) Rename spec constants from the current input module that are
    //   conflicting with functions currently in the combined module.
    updateSymbolAndAllUses<SpecConstantOp>(moduleClone, combinedModule,
                                           nextConflictID);
    updateSymbolAndAllUses<SpecConstantCompositeOp>(moduleClone, combinedModule,
                                                    nextConflictID);
    //   (2) Rename spec constants currently in the combined module that are
    //   conflicting with functions in the current input module.
    updateSymbolAndAllUses<SpecConstantOp>(combinedModule, moduleClone,
                                           nextConflictID);
    updateSymbolAndAllUses<SpecConstantCompositeOp>(combinedModule, moduleClone,
                                                    nextConflictID);

    // Rename function in the current input modules that are conflicting with
    // functions already in the combined module.
    updateSymbolAndAllUses<FuncOp>(moduleClone, combinedModule, nextConflictID);

    // Clone all the module's ops to the combined module.
    for (auto &op : moduleClone.getOps()) {
      if (dyn_cast<spirv::ModuleEndOp>(op))
        continue;

      combinedModuleBuilder.insert(op.clone());
    }
  }
}

} // namespace spirv
} // namespace mlir
