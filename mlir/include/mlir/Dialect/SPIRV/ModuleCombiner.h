#ifndef MLIR_DIALECT_SPIRV_MODULE_COMBINER_H_
#define MLIR_DIALECT_SPIRV_MODULE_COMBINER_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;

namespace spirv {
class ModuleOp;

/// To combine a number of MLIR spv modules, we move all the module-level ops
/// from all the input modules into one big combined module. To that end, the
/// combination process can proceed in 2 phases:
///
///   (1) resolving conflicts between pairs of ops from different modules
///   (2) deduplicate equivalent ops/sub-ops in the merged module. (TODO)
///
/// For the conflict resolution phase, the following rules are employed to
/// resolve such conflicts:
///
///   =========================================================================
///   FuncOp vs. FuncOp
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename one of the functions and update its refernces.
///   =========================================================================
///
///   =========================================================================
///   FuncOp vs. GlobalVariableOp
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename the global variable and update references to the renamed symbol.
///   =========================================================================
///
///   =========================================================================
///   FuncOp vs. SpecConstantOp
///   FuncOp vs. SpecConstantCompositeOp
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename the spec constant and update references to the renamed symbol.
///   =========================================================================
///
///   =========================================================================
///   GlobalVariableOp vs. GlobalVariableOp
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename either of the global variables and update references to it.
///   =========================================================================
///
///   =========================================================================
///   GlobalVariableOp vs. SpecConstantOp
///   FuncOp vs. SpecConstantCompositeOp
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename the global variable and update its references.
///   =========================================================================
///
///   =========================================================================
///   2 spec constants (scalar or composite)
///   Conflict: Same symbol name
///   -------------------------------------------------------------------------
///   Rename either of the constants and update its references.
///   =========================================================================
///
///   =========================================================================
///   EntryPointOp vs. EntryPointOp
///   Conflict: Same symbol name and execution model
///   -------------------------------------------------------------------------
///   No need to resolve this explicitly as it will be resolved as part of
///   resolving the conflict between the 2 associated functions.
///   =========================================================================
void combine(llvm::SmallVector<ModuleOp, 4> modules,
             OpBuilder &combinedModuleBuilder);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_MODULE_COMBINER_H_
