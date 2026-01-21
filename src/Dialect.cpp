#include "Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mini;

MiniDialect::MiniDialect(MLIRContext *context) 
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
    initialize();
}

void MiniDialect::initialize() {
    addOperations<ConstantOp, AddOp, SubOp, MulOp, DivOp, PrintOp>();
}

// ConstantOp Implementation
void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
    auto dataType = builder.getIntegerType(32);
    auto dataAttribute = builder.getI32IntegerAttr(static_cast<int32_t>(value));
    state.addAttribute("value", dataAttribute);
    state.addTypes(dataType);
}

double ConstantOp::getValue() {
    return (*this)->getAttrOfType<IntegerAttr>("value").getInt();
}

void ConstantOp::print(OpAsmPrinter &p) {
    p << " ";
    p << getValue();
    p << " : ";
    p << (*this)->getResult(0).getType();
}

// AddOp Implementation
void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void AddOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << ", ";
    p << (*this)->getOperand(1);
    p << " : ";
    p << (*this)->getResult(0).getType();
}

// SubOp Implementation
void SubOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void SubOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << ", ";
    p << (*this)->getOperand(1);
    p << " : ";
    p << (*this)->getResult(0).getType();
}

// MulOp Implementation
void MulOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void MulOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << ", ";
    p << (*this)->getOperand(1);
    p << " : ";
    p << (*this)->getResult(0).getType();
}

// DivOp Implementation
void DivOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void DivOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << ", ";
    p << (*this)->getOperand(1);
    p << " : ";
    p << (*this)->getResult(0).getType();
}

// PrintOp Implementation
void PrintOp::build(OpBuilder &builder, OperationState &state, Value input) {
    state.addOperands(input);
}

void PrintOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << " : ";
    p << (*this)->getOperand(0).getType();
}
