#include "MLIRGen.h"
#include "Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <iostream>

using namespace mlir;
using namespace mlir::mini;

MLIRGenerator::MLIRGenerator(MLIRContext &ctx) 
    : context(ctx), builder(&context) {
    // Start with global scope for top-level statements
    scopeStack.push_back({});
}

void MLIRGenerator::pushScope() {
    scopeStack.push_back({});
}

void MLIRGenerator::popScope() {
    if (scopeStack.size() > 1) {
        scopeStack.pop_back();
    }
}

void MLIRGenerator::defineVariable(const std::string& name, Value val) {
    if (scopeStack.empty()) {
        std::cerr << "ERROR: No scope available" << std::endl;
        return;
    }
    // In SSA, redefinition creates a new version
    scopeStack.back()[name] = val;
}

Value MLIRGenerator::lookupVariable(const std::string& name) {
    // Search from innermost to outermost scope
    for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    return nullptr;
}

Value MLIRGenerator::mlirGen(ExprAST &expr) {
    if (auto *numExpr = dynamic_cast<NumberExprAST*>(&expr)) {
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<ConstantOp>(loc, numExpr->value);
        return op->getResult(0);
    }
    
    if (auto *varRef = dynamic_cast<VarRefAST*>(&expr)) {
        Value val = lookupVariable(varRef->name);
        if (!val) {
            std::cerr << "ERROR: Undefined variable '" << varRef->name << "'" << std::endl;
        }
        return val;
    }
    
    if (auto *addExpr = dynamic_cast<AddExprAST*>(&expr)) {
        Value left = mlirGen(*addExpr->left);
        Value right = mlirGen(*addExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<AddOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *subExpr = dynamic_cast<SubExprAST*>(&expr)) {
        Value left = mlirGen(*subExpr->left);
        Value right = mlirGen(*subExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<SubOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *mulExpr = dynamic_cast<MulExprAST*>(&expr)) {
        Value left = mlirGen(*mulExpr->left);
        Value right = mlirGen(*mulExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<MulOp>(loc, left, right);
        return op->getResult(0);
    }
    
    if (auto *divExpr = dynamic_cast<DivExprAST*>(&expr)) {
        Value left = mlirGen(*divExpr->left);
        Value right = mlirGen(*divExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<DivOp>(loc, left, right);
        return op->getResult(0);
    }
    
    // Comparison operations
    if (auto *cmpExpr = dynamic_cast<CmpLTExprAST*>(&expr)) {
        Value left = mlirGen(*cmpExpr->left);
        Value right = mlirGen(*cmpExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<CmpOp>(loc, "slt", left, right);
        return op->getResult(0);
    }
    
    if (auto *cmpExpr = dynamic_cast<CmpGTExprAST*>(&expr)) {
        Value left = mlirGen(*cmpExpr->left);
        Value right = mlirGen(*cmpExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<CmpOp>(loc, "sgt", left, right);
        return op->getResult(0);
    }
    
    if (auto *cmpExpr = dynamic_cast<CmpEQExprAST*>(&expr)) {
        Value left = mlirGen(*cmpExpr->left);
        Value right = mlirGen(*cmpExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<CmpOp>(loc, "eq", left, right);
        return op->getResult(0);
    }
    
    if (auto *cmpExpr = dynamic_cast<CmpNEQExprAST*>(&expr)) {
        Value left = mlirGen(*cmpExpr->left);
        Value right = mlirGen(*cmpExpr->right);
        if (!left || !right) return nullptr;
        
        auto loc = builder.getUnknownLoc();
        auto op = builder.create<CmpOp>(loc, "ne", left, right);
        return op->getResult(0);
    }
    
    if (auto *callExpr = dynamic_cast<CallExprAST*>(&expr)) {
        // Look up function
        auto it = functionTable.find(callExpr->callee);
        if (it == functionTable.end()) {
            std::cerr << "ERROR: Undefined function '" << callExpr->callee << "'" << std::endl;
            return nullptr;
        }
        
        mini::FuncOp callee = it->second;
        
        // Check argument count
        if (callExpr->args.size() != callee.getNumArguments()) {
            std::cerr << "ERROR: Function '" << callExpr->callee << "' expects " 
                      << callee.getNumArguments() << " arguments but got " 
                      << callExpr->args.size() << std::endl;
            return nullptr;
        }
        
        // Generate argument values
        SmallVector<Value, 4> args;
        for (auto &arg : callExpr->args) {
            Value argValue = mlirGen(*arg);
            if (!argValue) return nullptr;
            args.push_back(argValue);
        }
        
        // Create call
        auto loc = builder.getUnknownLoc();
        auto callOp = builder.create<mini::CallOp>(
            loc, callExpr->callee, args, builder.getI32Type());
        
        return callOp->getResult(0);
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
    if (auto *returnStmt = dynamic_cast<ReturnStmtAST*>(&stmt)) {
        return mlirGen(*returnStmt);
    }
    if (auto *ifStmt = dynamic_cast<IfStmtAST*>(&stmt)) {
        return mlirGen(*ifStmt);
    }
    if (auto *whileStmt = dynamic_cast<WhileStmtAST*>(&stmt)) {
        return mlirGen(*whileStmt);
    }
    if (auto *exprStmt = dynamic_cast<ExprStmtAST*>(&stmt)) {
        // Just evaluate the expression and discard the result
        Value result = mlirGen(*exprStmt->expr);
        if (!result) return failure();
        return success();
    }
    
    std::cerr << "Unknown statement type" << std::endl;
    return failure();
}

LogicalResult MLIRGenerator::mlirGen(PrintStmtAST &stmt) {
    auto loc = builder.getUnknownLoc();
    Value exprValue = mlirGen(*stmt.expr);
    if (!exprValue) return failure();
    
    builder.create<PrintOp>(loc, exprValue);
    return success();
}

LogicalResult MLIRGenerator::mlirGen(VarDeclAST &stmt) {
    Value value = mlirGen(*stmt.value);
    if (!value) return failure();
    
    // In SSA, this creates a new version of the variable
    defineVariable(stmt.name, value);
    return success();
}

LogicalResult MLIRGenerator::mlirGen(ReturnStmtAST &stmt) {
    Value returnValue = mlirGen(*stmt.value);
    if (!returnValue) return failure();
    
    auto loc = builder.getUnknownLoc();
    builder.create<mini::ReturnOp>(loc, returnValue);
    return success();
}

LogicalResult MLIRGenerator::mlirGen(IfStmtAST &stmt) {
    auto loc = builder.getUnknownLoc();
    
    Value condition = mlirGen(*stmt.condition);
    if (!condition) {
        return failure();
    }
    
    // Se la condizione è i32, convertila in i1 confrontando con 0
    if (condition.getType().isInteger(32)) {
        auto zero = builder.create<ConstantOp>(loc, 0.0);
        auto cmpOp = builder.create<CmpOp>(loc, "ne", condition, zero->getResult(0));
        condition = cmpOp->getResult(0);
    }
    
    auto ifOp = builder.create<scf::IfOp>(
        loc,
        TypeRange{},
        condition,
        !stmt.elseBody.empty()
    );
    
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Generate then block
    for (auto &s : stmt.thenBody) {
        if (failed(mlirGen(*s))) {
            return failure();
        }
    }
    
    if (!stmt.elseBody.empty()) {
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        
        // Generate else block
        for (auto &s : stmt.elseBody) {
            if (failed(mlirGen(*s))) {
                return failure();
            }
        }
    }
    
    return success();
}

LogicalResult MLIRGenerator::mlirGen(WhileStmtAST &stmt) {
    auto loc = builder.getUnknownLoc();
    
    // Collect all variables that might be modified in the loop
    // For now, we'll track all variables in current scope
    std::vector<std::string> loopVars;
    std::vector<Value> initValues;
    std::vector<Type> loopVarTypes;
    
    for (const auto &var : scopeStack.back()) {
        loopVars.push_back(var.first);
        initValues.push_back(var.second);
        loopVarTypes.push_back(var.second.getType());
    }
    
    auto whileOp = builder.create<scf::WhileOp>(loc, loopVarTypes, initValues);
    
    // Before region - evaluate condition
    OpBuilder::InsertionGuard guard(builder);
    Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
    
    // Add block arguments for loop-carried values
    for (Type type : loopVarTypes) {
        beforeBlock->addArgument(type, loc);
    }
    
    builder.setInsertionPointToStart(beforeBlock);
    
    // Update scope with block arguments
    pushScope();
    for (size_t i = 0; i < loopVars.size(); i++) {
        defineVariable(loopVars[i], beforeBlock->getArgument(i));
    }
    
    Value condition = mlirGen(*stmt.condition);
    if (!condition) {
        popScope();
        return failure();
    }
    
    // Convert i32 to i1 if necessary
    if (condition.getType().isInteger(32)) {
        auto zero = builder.create<ConstantOp>(loc, 0.0);
        auto cmpOp = builder.create<CmpOp>(loc, "ne", condition, zero->getResult(0));
        condition = cmpOp->getResult(0);
    }
    
    // Pass current values to after region
    SmallVector<Value> beforeYieldValues;
    for (size_t i = 0; i < loopVars.size(); i++) {
        beforeYieldValues.push_back(lookupVariable(loopVars[i]));
    }
    
    builder.create<scf::ConditionOp>(loc, condition, beforeYieldValues);
    popScope();
    
    // After region - loop body
    Block *afterBlock = builder.createBlock(&whileOp.getAfter());
    
    // Add block arguments for loop-carried values
    for (Type type : loopVarTypes) {
        afterBlock->addArgument(type, loc);
    }
    
    builder.setInsertionPointToStart(afterBlock);
    
    // Update scope with block arguments
    pushScope();
    for (size_t i = 0; i < loopVars.size(); i++) {
        defineVariable(loopVars[i], afterBlock->getArgument(i));
    }
    
    for (auto &s : stmt.body) {
        if (failed(mlirGen(*s))) {
            popScope();
            return failure();
        }
    }
    
    // Yield updated values back to before region
    SmallVector<Value> afterYieldValues;
    for (size_t i = 0; i < loopVars.size(); i++) {
        afterYieldValues.push_back(lookupVariable(loopVars[i]));
    }
    
    builder.create<scf::YieldOp>(loc, afterYieldValues);
    popScope();
    
    // Update outer scope with final values
    for (size_t i = 0; i < loopVars.size(); i++) {
        defineVariable(loopVars[i], whileOp.getResult(i));
    }
    
    return success();
}

LogicalResult MLIRGenerator::mlirGen(FunctionAST &func) {
    // Enter function scope
    pushScope();
    
    auto loc = builder.getUnknownLoc();
    
    // Build function type (all i32)
    SmallVector<Type, 4> argTypes(func.params.size(), builder.getI32Type());
    auto funcType = builder.getFunctionType(argTypes, builder.getI32Type());
    
    // Create function
    auto funcOp = builder.create<mini::FuncOp>(loc, func.name, funcType);
    functionTable[func.name] = funcOp;
    
    // Create entry block with arguments
    Block *entryBlock = builder.createBlock(&funcOp.getBody());
    for (auto argType : argTypes) {
        entryBlock->addArgument(argType, loc);
    }
    
    builder.setInsertionPointToStart(entryBlock);
    
    // Add parameters to scope
    for (size_t i = 0; i < func.params.size(); i++) {
        defineVariable(func.params[i], entryBlock->getArgument(i));
    }
    
    // Generate function body
    bool hasReturn = false;
    for (auto &stmt : func.body) {
        if (hasReturn) {
            std::cerr << "WARNING: Unreachable code after return" << std::endl;
            break;
        }
        
        if (dynamic_cast<ReturnStmtAST*>(stmt.get())) {
            hasReturn = true;
        }
        
        if (failed(mlirGen(*stmt))) {
            popScope();
            return failure();
        }
    }
    
    // Add implicit return if needed
    if (!hasReturn) {
        auto zero = builder.create<ConstantOp>(loc, 0.0);
        builder.create<mini::ReturnOp>(loc, zero->getResult(0));
    }
    
    // Exit function scope
    popScope();
    
    // Reset insertion point to end of module
    builder.setInsertionPointToEnd(&funcOp->getParentRegion()->front());
    
    return success();
}

ModuleOp MLIRGenerator::mlirGen(ModuleAST &moduleAST) {
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    // Generate all functions
    for (auto &func : moduleAST.functions) {
        if (failed(mlirGen(*func))) {
            module.emitError("Failed to generate function");
            return nullptr;
        }
    }
    
    // Generate main function for top-level statements
    if (!moduleAST.statements.empty()) {
        if (functionTable.find("main") != functionTable.end()) {
            module.emitError("Cannot have top-level statements with explicit main");
            return nullptr;
        }
        
        auto mainType = builder.getFunctionType({}, builder.getI32Type());
        auto mainFunc = builder.create<mini::FuncOp>(loc, "main", mainType);
        functionTable["main"] = mainFunc;
        
        Block *mainBlock = builder.createBlock(&mainFunc.getBody());
        builder.setInsertionPointToStart(mainBlock);
        
        // Generate statements (global scope already exists)
        for (auto &stmt : moduleAST.statements) {
            if (failed(mlirGen(*stmt))) {
                module.emitError("Failed to generate statement");
                return nullptr;
            }
        }
        
        // Return 0 from main
        auto zero = builder.create<ConstantOp>(loc, 0.0);
        builder.create<mini::ReturnOp>(loc, zero->getResult(0));
    }
    
    if (failed(verify(module))) {
        module.emitError("Module verification failed");
        return nullptr;
    }
    
    return module;
}
