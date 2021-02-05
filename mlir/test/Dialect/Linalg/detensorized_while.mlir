// RUN: mlir-opt %s -linalg-detensorize -canonicalize | FileCheck %s

func @main() -> tensor<i32> attributes {iree.module.export} {
  %cst = constant dense<1> : tensor<i32>
  %cst_0 = constant dense<3> : tensor<i32>
  br ^bb1(%cst : tensor<i32>)
^bb1(%0: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %1 = linalg.init_tensor [] : tensor<i1>
  %2 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%0, %cst_0 : tensor<i32>, tensor<i32>) outs(%1 : tensor<i1>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  // no predecessors
    %8 = cmpi slt, %arg0, %arg1 : i32
    linalg.yield %8 : i1
  } -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  cond_br %3, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)
^bb2(%4: tensor<i32>):  // pred: ^bb1
  %5 = linalg.init_tensor [] : tensor<i32>
  %6 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%4, %4 : tensor<i32>, tensor<i32>) outs(%5 : tensor<i32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
    %8 = addi %arg0, %arg1 : i32
    linalg.yield %8 : i32
  } -> tensor<i32>
  br ^bb1(%6 : tensor<i32>)
^bb3(%7: tensor<i32>):  // pred: ^bb1
  return %7 : tensor<i32>
}
// CHECK-LABEL: func @main() -> tensor<i32>
// CHECK:         %[[c1:.*]] = constant dense<1>
// CHECK:         %[[c3:.*]] = constant 3
// CHECK:         br ^bb1(%[[c1]] : tensor<i32>)
// CHECK:       ^[[bb1:.*]](%[[bb1_arg:.*]]: tensor<i32>)
// CHECK:         tensor.extract %[[bb1_arg]][]
// CHECK:         %[[cmp_res:.*]] = cmpi slt
// CHECK:         cond_br %[[cmp_res]]
// CHECK:       ^[[bb2:.*]](%[[bb2_arg:.*]]: tensor<i32>)
// CHECK:         tensor.extract
// CHECK:         tensor.extract
// CHECK:         %[[add_res:.*]] = addi
// CHECK:         %[[fe_res:.*]] = tensor.from_elements %[[add_res]]
// CHECK:         %[[res:.*]] = linalg.tensor_reshape %[[fe_res]] []
// CHECK:         br ^bb1(%[[res]] : tensor<i32>)
