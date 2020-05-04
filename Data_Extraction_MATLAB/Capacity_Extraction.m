counter = 0
for i = 1:length(B0005.cycle) 
    if strcmp(B0005.cycle(i).type, 'discharge')
        counter = counter+1;
        capacity(counter) = B0005.cycle(i).data.Capacity;
    end
end
