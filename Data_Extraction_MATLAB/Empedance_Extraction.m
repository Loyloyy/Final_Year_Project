counter = 0
for i = 1:length(B0005.cycle) 
    if strcmp(B0005.cycle(i).type, 'impedance')
        counter = counter+1;
        Re(counter) = B0005.cycle(i).data.Re;
        Rct(counter) = B0005.cycle(i).data.Rct;
    end
end
