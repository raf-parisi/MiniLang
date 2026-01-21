#include "MLIRGen.h"
#include "Dialect.h"
#include "mlir/IR/Verifier.h"
#include <iostream>

using namespace mlir;
using namespace mlir::mini;

MLIRGenerator::MLIRGenerator(MLIRContext &ctx) 
    : context(ctx), builder(&context) {}

Value MLIRGenerator::mlirGen(ExprAST &expr) {
    std::cout << "  Generating MLIR for expression..." << std::endl;
    
    if (auto *numExpr = dynamic_cast<NumberExprAST*>(&expr)) {
        std::cout << "    Creating constant: " << numExpr->value << std::endl;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<ConstantOp>(loc, numExpr->value);
        return op->getResult(0);
    }
    
    if (auto *varRef = dynamic_cast<VarRefAST*>(&expr)) {
        std::cout << "    Looking up variable: " << varRef->name << std::endl;
        auto it = symbolTable.find(varRef->name);
        if (it == symbolTable.end()) {
            std::cerr << "ERROR: Undefined variable '" << varRef->name << "'" << std::endl;
            return nullptr;
        }
        std::cout << "    Variable found" << std::endl;
        return it->second;
    }
    
    if (auto *addExpr = dynamic_cast<AddExprAST*>(&expr)) {
        std::cout << "    Creating add operation" << std::endl;
        Value left = mlirGen(*addExpr->left);
        Value right = mlirGen(*addExpr->right);
        if (!left || !right) return nullptr;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<AddOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *subExpr = dynamic_cast<SubExprAST*>(&expr)) {
        std::cout << "    Creating sub operation" << std::endl;
        Value left = mlirGen(*subExpr->left);
        Value right = mlirGen(*subExpr->right);
        if (!left || !right) return nullptr;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<SubOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *mulExpr = dynamic_cast<MulExprAST*>(&expr)) {
        std::cout << "    Creating mul operation" << std::endl;
        Value left = mlirGen(*mulExpr->left);
        Value right = mlirGen(*mulExpr->right);
        if (!left || !right) return nullptr;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<MulOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *divExpr = dynamic_cast<DivExprAST*>(&expr)) {
        std::cout << "    Creating div operation" << std::endl;
        Value left = mlirGen(*divExpr->left);
        Value right = mlirGen(*divExpr->right);
        if (!left || !right) return nullptr;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<DivOp>(loc, left, right);
        return op->getResult(0);
    }
    
    std::cerr << "Unknown expression type" << std::endl;
    return nullptr;
}

LogicalResult MLIRGenerator::mlirGen(StmtAST &stmt) {
    if (auto *printStmt = dynamic_cast<PrintStmtAST*>(&stmt)) {
        return mlirGen(*printStmt);
    }
    
    if (auto *varDecl = dynamic_cast<VarDeclAST*>(&stmt)) {
        return mlirGen(*varDecl);
    }
    
    std::cerr << "Unknown statement type" << std::endl;
    return failure();
}

LogicalResult MLIRGenerator::mlirGen(PrintStmtAST &stmt) {
    std::cout << "  Generating print statement" << std::endl;
    auto loc = builder.getUnknownLoc();
    
    Value exprValue = mlirGen(*stmt.expr);
    if (!exprValue) {
        return failure();
    }
    
    builder.create<PrintOp>(loc, exprValue);
    return success();
}

LogicalResult MLIRGenerator::mlirGen(VarDeclAST &stmt) {
    std::cout << "  Generating variable declaration: " << stmt.name << std::endl;
    
    Value value = mlirGen(*stmt.value);
    if (!value) {
        return failure();
    }
    
    // Store in symbol table
    symbolTable[stmt.name] = value;
    std::cout << "    Variable '" << stmt.name << "' stored in symbol table" << std::endl;
    
    return success();
}

ModuleOp MLIRGenerator::mlirGen(ModuleAST &moduleAST) {
    std::cout << "Creating module..." << std::endl;
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    std::cout << "Module created" << std::endl;
    
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    std::cout << "Processing " << moduleAST.statements.size() << " statements" << std::endl;
    
    for (size_t i = 0; i < moduleAST.statements.size(); ++i) {
        std::cout << "Processing statement " << (i+1) << std::endl;
        if (failed(mlirGen(*moduleAST.statements[i]))) {
            module.emitError("Failed to generate MLIR for statement");
            return nullptr;
        }
    }
    
    std::cout << "Verifying module..." << std::endl;
    if (failed(verify(module))) {
        module.emitError("Module verification failed");
        return nullptr;
    }
    std::cout << "Module verified" << std::endl;
    
    return module;
}
