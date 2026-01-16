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
    //static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &p);
    //LogicalResult verify();
};

class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.add"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs);
    //static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &p);
    //LogicalResult verify();
};

class PrintOp : public Op<PrintOp, OpTrait::OneOperand, OpTrait::ZeroResults> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "mini.print"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
    static void build(OpBuilder &builder, OperationState &state, Value input);
    //static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &p);
    //LogicalResult verify();
};

} // namespace mini
} // namespace mlir

#endif 
