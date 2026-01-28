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

// Comparison expressions
class CmpLTExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    CmpLTExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class CmpGTExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    CmpGTExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class CmpEQExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    CmpEQExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class CmpNEQExprAST : public ExprAST {
public:
    std::unique_ptr<ExprAST> left;
    std::unique_ptr<ExprAST> right;
    
    CmpNEQExprAST(std::unique_ptr<ExprAST> l, std::unique_ptr<ExprAST> r)
        : left(std::move(l)), right(std::move(r)) {}
};

class CallExprAST : public ExprAST {
public:
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;
    
    CallExprAST(std::string name, std::vector<std::unique_ptr<ExprAST>> arguments)
        : callee(std::move(name)), args(std::move(arguments)) {}
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

class ReturnStmtAST : public StmtAST {
public:
    std::unique_ptr<ExprAST> value;
    
    ReturnStmtAST(std::unique_ptr<ExprAST> v) : value(std::move(v)) {}
};

class ExprStmtAST : public StmtAST {
public:
    std::unique_ptr<ExprAST> expr;
    
    ExprStmtAST(std::unique_ptr<ExprAST> e) : expr(std::move(e)) {}
};

class IfStmtAST : public StmtAST {
public:
    std::unique_ptr<ExprAST> condition;
    std::vector<std::unique_ptr<StmtAST>> thenBody;
    std::vector<std::unique_ptr<StmtAST>> elseBody;
    
    IfStmtAST(std::unique_ptr<ExprAST> cond,
              std::vector<std::unique_ptr<StmtAST>> thenStmts,
              std::vector<std::unique_ptr<StmtAST>> elseStmts = {})
        : condition(std::move(cond)),
          thenBody(std::move(thenStmts)),
          elseBody(std::move(elseStmts)) {}
};

class FunctionAST {
public:
    std::string name;
    std::vector<std::string> params;
    std::vector<std::unique_ptr<StmtAST>> body;
    
    FunctionAST(std::string n, std::vector<std::string> p, std::vector<std::unique_ptr<StmtAST>> b)
        : name(std::move(n)), params(std::move(p)), body(std::move(b)) {}
};

class ModuleAST {
public:
    std::vector<std::unique_ptr<FunctionAST>> functions;
    std::vector<std::unique_ptr<StmtAST>> statements;
    
    void addFunction(std::unique_ptr<FunctionAST> func) {
        functions.push_back(std::move(func));
    }
    
    void addStatement(std::unique_ptr<StmtAST> stmt) {
        statements.push_back(std::move(stmt));
    }
};

#endif
