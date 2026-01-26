#ifndef DIALECT_H
#define DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace mini {

class MiniDialect : public Dialect {
public:
    explicit MiniDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "mini"; }
    void initialize();
};

class ConstantOp : public Op<ConstantOp, OpTrait::ZeroOperands, OpTrait::OneResult, OpTrait::ConstantLike> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.constant"; }
    static constexpr ::llvm::StringLiteral getAttributeNameValue() { return ::llvm::StringLiteral("value"); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        static ::llvm::StringRef attrNames[] = {getAttributeNameValue()};
        return ::llvm::ArrayRef(attrNames);
    }
    static void build(OpBuilder &builder, OperationState &state, double value);
    double getValue();
    void print(OpAsmPrinter &p);
};

class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.add"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class SubOp : public Op<SubOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.sub"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class MulOp : public Op<MulOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.mul"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class DivOp : public Op<DivOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.div"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class CmpOp : public Op<CmpOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.cmp"; }
    
    static constexpr ::llvm::StringLiteral getAttributeNamePredicate() { 
        return ::llvm::StringLiteral("predicate"); 
    }
    
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        static ::llvm::StringRef attrNames[] = {getAttributeNamePredicate()};
        return ::llvm::ArrayRef(attrNames);
    }
    
    static void build(OpBuilder &builder, OperationState &state, 
                     StringRef predicate, Value lhs, Value rhs);
    
    StringRef getPredicate();
    void print(OpAsmPrinter &p);
};

class PrintOp : public Op<PrintOp, OpTrait::OneOperand, OpTrait::ZeroResults> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.print"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value input);
    void print(OpAsmPrinter &p);
};

class FuncOp : public Op<FuncOp, 
    OpTrait::ZeroOperands,
    OpTrait::ZeroResults,
    OpTrait::OneRegion,
    OpTrait::IsIsolatedFromAbove,
    SymbolOpInterface::Trait> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.func"; }
    
    static constexpr ::llvm::StringLiteral getSymNameAttrName() { 
        return ::llvm::StringLiteral("sym_name"); 
    }
    static constexpr ::llvm::StringLiteral getFunctionTypeAttrName() { 
        return ::llvm::StringLiteral("function_type"); 
    }
    
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        static ::llvm::StringRef attrNames[] = {getSymNameAttrName(), getFunctionTypeAttrName()};
        return ::llvm::ArrayRef(attrNames);
    }
    
    static void build(OpBuilder &builder, OperationState &state, 
                     StringRef name, FunctionType type);
    
    StringRef getName() { 
        return (*this)->getAttrOfType<StringAttr>(getSymNameAttrName()).getValue(); 
    }
    
    FunctionType getFunctionType() {
        return (*this)->getAttrOfType<TypeAttr>(getFunctionTypeAttrName()).getValue().cast<FunctionType>();
    }
    
    Region &getBody() { return (*this)->getRegion(0); }
    unsigned getNumArguments() { return getFunctionType().getNumInputs(); }
    
    void print(OpAsmPrinter &p);
};

class CallOp : public Op<CallOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.call"; }
    
    static constexpr ::llvm::StringLiteral getCalleeAttrName() { 
        return ::llvm::StringLiteral("callee"); 
    }
    
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        static ::llvm::StringRef attrNames[] = {getCalleeAttrName()};
        return ::llvm::ArrayRef(attrNames);
    }
    
    static void build(OpBuilder &builder, OperationState &state,
                     StringRef callee, ArrayRef<Value> operands, Type resultType);
    
    StringRef getCallee() { 
        return (*this)->getAttrOfType<FlatSymbolRefAttr>(getCalleeAttrName()).getValue(); 
    }
    
    void print(OpAsmPrinter &p);
};

class ReturnOp : public Op<ReturnOp, OpTrait::OneOperand, OpTrait::ZeroResults, OpTrait::IsTerminator> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.return"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    static void build(OpBuilder &builder, OperationState &state, Value operand);
    void print(OpAsmPrinter &p);
};

} // namespace mini
} // namespace mlir

#endif
