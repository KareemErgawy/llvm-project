// TODO This is just to aid in development, will be properly updated as a test later.

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {}
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.Branch ^header

  ^header:
    spv.mlir.structured_branch "Loop" ^entry [^merge, ^continue]

  ^entry:
    %val0 = spv.Load "Function" %var : i32
    %cmp = spv.SLessThan %val0, %count : i32
    spv.BranchConditional %cmp [1, 1], ^body, ^merge

  ^body:
    // Do nothing
    spv.Branch ^continue

  ^continue:
    %val1 = spv.Load "Function" %var : i32
    %add = spv.IAdd %val1, %one : i32
    spv.Store "Function" %var, %add : i32
    spv.Branch ^header

  ^merge:
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}

// CHECK:        spv.Constant 0 : i32
// CHECK-NEXT:   spv.Variable init
// CHECK-NEXT:   spv.Branch ^[[loop:.*]]

// CHECK-NEXT: ^[[loop]]:
// CHECK-NEXT:   spv.mlir.loop
// CHECK-NEXT:     spv.Branch ^[[bb1:.*]]

// CHECK-NEXT:   ^[[bb1:.*]]:
// CHECK-NEXT:     spv.Branch ^[[header:.*]]

// CHECK-NEXT:   ^[[header]]:
// CHECK-NEXT:     spv.Load
// CHECK-NEXT:     spv.SLessThan
// CHECK-NEXT:     spv.BranchConditional %{{.*}} [1, 1], ^[[bb2:.*]], ^[[merge:.*]]

// CHECK-NEXT:   ^[[bb2]]:
// CHECK-NEXT:     spv.Branch ^[[continue:.*]]

// CHECK-NEXT:   ^[[continue]]:
// CHECK-NEXT:     spv.Load
// CHECK-NEXT:     spv.Constant 1
// CHECK-NEXT:     spv.IAdd
// CHECK-NEXT:     spv.Store
// CHECK-NEXT:     spv.Branch ^[[header]]

// CHECK-NEXT:   ^[[merge]]:
// CHECK-NEXT:     spv.mlir.merge
