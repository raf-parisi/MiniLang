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
    addOperations<ConstantOp, AddOp, SubOp, MulOp, DivOp, CmpOp, PrintOp, 
                  FuncOp, CallOp, ReturnOp>();
}

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
    p << " " << getValue() << " : " << (*this)->getResult(0).getType();
}

void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void AddOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << ", " << (*this)->getOperand(1) 
      << " : " << (*this)->getResult(0).getType();
}

void SubOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void SubOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << ", " << (*this)->getOperand(1) 
      << " : " << (*this)->getResult(0).getType();
}

void MulOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void MulOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << ", " << (*this)->getOperand(1) 
      << " : " << (*this)->getResult(0).getType();
}

void DivOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

void DivOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << ", " << (*this)->getOperand(1) 
      << " : " << (*this)->getResult(0).getType();
}

void CmpOp::build(OpBuilder &builder, OperationState &state, 
                  StringRef predicate, Value lhs, Value rhs) {
    state.addAttribute(getAttributeNamePredicate(), builder.getStringAttr(predicate));
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getI1Type());
}

StringRef CmpOp::getPredicate() {
    return (*this)->getAttrOfType<StringAttr>(getAttributeNamePredicate()).getValue();
}

void CmpOp::print(OpAsmPrinter &p) {
    p << " \"" << getPredicate() << "\", " 
      << (*this)->getOperand(0) << ", " << (*this)->getOperand(1)
      << " : " << (*this)->getResult(0).getType();
}

void PrintOp::build(OpBuilder &builder, OperationState &state, Value input) {
    state.addOperands(input);
}

void PrintOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << " : " << (*this)->getOperand(0).getType();
}

void FuncOp::build(OpBuilder &builder, OperationState &state, 
                   StringRef name, FunctionType type) {
    state.addAttribute(getSymNameAttrName(), builder.getStringAttr(name));
    state.addAttribute(getFunctionTypeAttrName(), TypeAttr::get(type));
    state.addRegion();
}

void FuncOp::print(OpAsmPrinter &p) {
    p << " @" << getName() << "(";
    FunctionType fnType = getFunctionType();
    llvm::interleaveComma(fnType.getInputs(), p, [&](Type type) { p << type; });
    p << ") -> ";
    if (fnType.getNumResults() == 1) {
        p << fnType.getResult(0);
    }
    if (!getBody().empty()) {
        p << " ";
        p.printRegion(getBody(), false);
    }
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   StringRef callee, ArrayRef<Value> operands, Type resultType) {
    state.addAttribute(getCalleeAttrName(), 
                      SymbolRefAttr::get(builder.getContext(), callee));
    state.addOperands(operands);
    state.addTypes(resultType);
}

void CallOp::print(OpAsmPrinter &p) {
    p << " @" << getCallee() << "(";
    p.printOperands((*this)->getOperands());
    p << ") : (";
    llvm::interleaveComma((*this)->getOperandTypes(), p);
    p << ") -> " << (*this)->getResult(0).getType();
}

void ReturnOp::build(OpBuilder &builder, OperationState &state, Value operand) {
    state.addOperands(operand);
}

void ReturnOp::print(OpAsmPrinter &p) {
    p << " " << (*this)->getOperand(0) << " : " << (*this)->getOperand(0).getType();
}
