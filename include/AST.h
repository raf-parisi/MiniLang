#ifndef AST_H
#define AST_H

#include <memory>
#include <vector>
#include <string>

class ExprAST {
public:
    virtual ~ExprAST() = default;
};

class NumberExprAST : public ExprAST {
public:
    double value;
    NumberExprAST(double val) : value(val) {}
};

class VarRefAST : public ExprAST {
public:
    std::string name;
    VarRefAST(std::string n) : name(std::move(n)) {}
};

class AddExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    AddExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class SubExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    SubExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class MulExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    MulExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class DivExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    DivExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

// Statement types
class StmtAST {
public:
    virtual ~StmtAST() = default;
};

class PrintStmtAST : public StmtAST {
public:
    std::unique_ptr<ExprAST> expr;
    
    PrintStmtAST(std::unique_ptr<ExprAST> e) : expr(std::move(e)) {}
};

class VarDeclAST : public StmtAST {
public:
    std::string name;
    std::unique_ptr<ExprAST> value;
    
    VarDeclAST(std::string n, std::unique_ptr<ExprAST> v)
        : name(std::move(n)), value(std::move(v)) {}
};

class ModuleAST {
public:
    std::vector<std::unique_ptr<StmtAST>> statements;
    
    void addStatement(std::unique_ptr<StmtAST> stmt) {
        statements.push_back(std::move(stmt));
    }
};

#endif
