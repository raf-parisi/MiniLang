#ifndef DIALECT_H
#define DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
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
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class SubOp : public Op<SubOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.sub"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class MulOp : public Op<MulOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.mul"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class DivOp : public Op<DivOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.div"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    void print(OpAsmPrinter &p);
};

class PrintOp : public Op<PrintOp, OpTrait::OneOperand, OpTrait::ZeroResults> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.print"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value input);
    void print(OpAsmPrinter &p);
};

} // namespace mini
} // namespace mlir

#endif
