#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <map>
#include <string>

namespace mlir {
namespace mini {

class MLIRGenerator {
private:
    MLIRContext &context;
    OpBuilder builder;
    ModuleOp module;
    std::map<std::string, Value> symbolTable;
    
    Value mlirGen(ExprAST &expr);
    LogicalResult mlirGen(StmtAST &stmt);
    LogicalResult mlirGen(PrintStmtAST &stmt);
    LogicalResult mlirGen(VarDeclAST &stmt);

public:
    MLIRGenerator(MLIRContext &ctx);
    ModuleOp mlirGen(ModuleAST &moduleAST);
};

} 
} 

#endif
