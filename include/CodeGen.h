#ifndef CODEGEN_H
#define CODEGEN_H

#include "llvm/IR/Module.h"
#include <string>

namespace mlir {
namespace mini {

bool compileModule(llvm::Module &module, const std::string &outputFile);

}
} 

#endif 
