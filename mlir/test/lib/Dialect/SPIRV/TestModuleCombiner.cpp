#include "mlir/Dialect/SPIRV/ModuleCombiner.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class TestModuleCombinerPass
    : public PassWrapper<TestModuleCombinerPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  TestModuleCombinerPass() = default;
  TestModuleCombinerPass(const TestModuleCombinerPass &) {}
  void runOnOperation() override;
};
} // namespace

void TestModuleCombinerPass::runOnOperation() {
  auto modules = llvm::to_vector<4>(getOperation().getOps<spirv::ModuleOp>());

  OpBuilder combinedModuleBuilder(modules[0]);
  spirv::combine(modules, combinedModuleBuilder);

  for (auto module : modules) {
    module.erase();
  }
}

namespace mlir {
void registerTestSpirvModuleCombinerPass() {
  PassRegistration<TestModuleCombinerPass> registration(
      "test-spirv-module-combiner", "Tests SPIR-V module combiner library");
}
} // namespace mlir
