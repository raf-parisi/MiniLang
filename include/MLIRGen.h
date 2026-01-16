#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>

namespace mlir {
namespace mini {

class MLIRGenerator {
private:
    MLIRContext &context;
    OpBuilder builder;
    ModuleOp module;
    
    Value mlirGen(ExprAST &expr);
    LogicalResult mlirGen(PrintStmtAST &stmt);

public:
    MLIRGenerator(MLIRContext &ctx);
    ModuleOp mlirGen(ModuleAST &moduleAST);
};

} 
} 

#endif 
