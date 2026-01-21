#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mini {

std::unique_ptr<Pass> createConstantFoldingPass();

} 
} 

#endif 
