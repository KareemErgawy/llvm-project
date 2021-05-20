//===- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_UTILS_H_
#define MLIR_TARGET_SPIRV_UTILS_H_

namespace mlir {
namespace spirv {
enum DeserializerConfig {
  // Deserialize SPIR-V structured loops to `spv.loop`.
  DESERIALIZE_TO_STRUCTURED_OPS,
  // Deserialize SPIR-V structured loops to `spv.structured_branch`.
  DESERIALIZE_TO_STRUCTURED_CONTROL_FLOW,
  // Deserialize SPIR-V structured loops to `spv.structured_branch` to convert
  // to `spv.loop`.
  DESERIALIZE_TO_STRUCTURED_CONTROL_FLOW_THEN_TO_STRUCTURED_OPS,
};
} // namespace spirv
} // namespace mlir

#endif // MLIR_TARGET_SPIRV_UTILS_H_
