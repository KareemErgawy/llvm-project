// RUN: mlir-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

func @loop(%count : i32) -> () {
  %zero = spv.Constant 0: i32
  %one = spv.Constant 1: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  spv.mlir.loop {
    spv.Branch ^header

  ^header:
    %val0 = spv.Load "Function" %var : i32
    %cmp = spv.SLessThan %val0, %count : i32
    spv.BranchConditional %cmp [1, 1], ^continue, ^merge

  ^continue:
    %val1 = spv.Load "Function" %var : i32
    %add = spv.IAdd %val1, %one : i32
    spv.Store "Function" %var, %add : i32
    spv.Branch ^header

  ^merge:
    spv.mlir.merge
  }
  spv.Return
}

// CHECK:    spv.mlir.structured_branch "Loop" ^bb2 [^bb5, ^bb3]
// CHECK:  ^bb2:
// CHECK:    %3 = spv.Load "Function" %2 : i32
// CHECK:    %4 = spv.SLessThan %3, %arg0 : i32
// CHECK:    spv.BranchConditional %4 [1, 1], ^bb3, ^bb5
// CHECK:  ^bb3:
// CHECK:    spv.Branch ^bb4
// CHECK:  ^bb4:
// CHECK:    %5 = spv.Load "Function" %2 : i32
// CHECK:    %6 = spv.IAdd %5, %1 : i32
// CHECK:    spv.Store "Function" %2, %6 : i32
// CHECK:    spv.Branch ^bb1
// CHECK:  ^bb5:
// CHECK:    spv.Branch ^bb6
