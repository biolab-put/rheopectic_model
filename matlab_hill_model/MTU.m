function d2lm = MTU(x, m, km,kt, c)
    F_ce=x(1);
    lm=x(2);
    dlm=x(3);
    d2lm= 1/m * (-abs(c)*dlm-abs(km)*lm-abs(kt)*lm+F_ce);
    
    %Uproszone k = km, kt i km polaczone w km
    %d2lm= 1/abs(m) * (-abs(c)*dlm-abs(km)*lm+F_ce);
end