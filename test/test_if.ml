x = 10;
y = 5;

if (x > y) {
    result = x + y;
    print result;
} else {
    result = x - y;
    print result;
}

a = 3;
b = 3;

if (a == b) {
    print 100;
}

if (a != b) {
    print 999;
} else {
    print 42;
}

fn compare(a,b){
    if (a > b){
        print a;
    } else{
        print b;
    }
    return 0;
}
