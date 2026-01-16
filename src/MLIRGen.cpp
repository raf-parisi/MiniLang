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
        std::cout << "    Constant created" << std::endl;
        return op->getResult(0);
    }
    
    if (auto *addExpr = dynamic_cast<AddExprAST*>(&expr)) {
        std::cout << "    Creating add operation" << std::endl;
        Value left = mlirGen(*addExpr->left);
        Value right = mlirGen(*addExpr->right);
        if (!left || !right) return nullptr;
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<AddOp>(loc, left, right);
        std::cout << "    Add created" << std::endl;
        return op->getResult(0);
    }
    
    std::cerr << "Unknown expression type" << std::endl;
    return nullptr;
}

ModuleOp MLIRGenerator::mlirGen(ModuleAST &moduleAST) {
    std::cout << "Creating module..." << std::endl;
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    std::cout << "Module created" << std::endl;
    
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    std::cout << "Builder insertion point set" << std::endl;
    
    std::cout << "Processing " << moduleAST.statements.size() << " statements" << std::endl;
    
    for (size_t i = 0; i < moduleAST.statements.size(); ++i) {
        std::cout << "Processing statement " << (i+1) << std::endl;
        Value exprValue = mlirGen(*moduleAST.statements[i]->expr);
        if (!exprValue) {
            module.emitError("Failed to generate MLIR for expression");
            return nullptr;
        }
        
        std::cout << "Creating print op" << std::endl;
        builder.create<PrintOp>(loc, exprValue);
        std::cout << "Print op created" << std::endl;
    }
    
    std::cout << "Verifying module..." << std::endl;
    if (failed(verify(module))) {
        module.emitError("Module verification failed");
        return nullptr;
    }
    std::cout << "Module verified" << std::endl;
    
    return module;
}
