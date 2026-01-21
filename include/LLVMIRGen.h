#ifndef LLVMIRGEN_H
#define LLVMIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include <memory>

namespace mlir {
namespace mini {

std::unique_ptr<llvm::Module> convertToLLVMIR(ModuleOp module, llvm::LLVMContext &llvmContext);

} // namespace mini
} // namespace mlir

#endif // LLVMIRGEN_H
