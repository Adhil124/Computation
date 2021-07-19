function x = Relu(y)
    t = y > 0;
    x = y.*t;
    
end