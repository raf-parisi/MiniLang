#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <map>
#include <vector>
#include <string>

namespace mlir {
namespace mini {

class FuncOp;

class MLIRGenerator {
private:
    MLIRContext &context;
    OpBuilder builder;
    ModuleOp module;
    
    // Function table
    std::map<std::string, mini::FuncOp> functionTable;
    
    // Scope stack - each scope maps variable names to SSA values
    std::vector<std::map<std::string, Value>> scopeStack;
    
    // Scope management
    void pushScope();
    void popScope();
    void defineVariable(const std::string& name, Value val);
    Value lookupVariable(const std::string& name);
    
    // Expression code generation
    Value mlirGen(ExprAST &expr);
    
    // Statement code generation
    LogicalResult mlirGen(StmtAST &stmt);
    LogicalResult mlirGen(PrintStmtAST &stmt);
    LogicalResult mlirGen(VarDeclAST &stmt);
    LogicalResult mlirGen(ReturnStmtAST &stmt);
    LogicalResult mlirGen(IfStmtAST &stmt);
    
    // Function code generation
    LogicalResult mlirGen(FunctionAST &func);

public:
    MLIRGenerator(MLIRContext &ctx);
    ModuleOp mlirGen(ModuleAST &moduleAST);
};

} 
} 

#endif
