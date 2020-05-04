function mydatagather
for i=1:1:2
t1=xlsread('S00418002151_1st Cycle_C01consxout.xlsx');
t2=xlsread('S00418002151_2nd Cycle_C01consxout.xlsx');
t3=xlsread('S00418002151_3rd Cycle_C01consxout.xlsx');
xgather=[t1;t2;t3];
end
xlswrite('xgather.xlsx',xgather);

end