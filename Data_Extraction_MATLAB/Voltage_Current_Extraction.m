counter = 0
fvoltage= 0
for i = 1:length(B0005.cycle) 
    if strcmp(B0005.cycle(i).type, 'discharge')
        counter = counter+1;
	j = 1
	k(counter) = 0
	voltage1 = 0
	voltage2 = 0
	voltage3 = 0
	tvoltage = 0
	fvoltage(counter) = 0
	for j = 1:length(B0005.cycle(i).data.Voltage_measured) - 1
            voltage1 = B0005.cycle(i).data.Voltage_measured(j);
	    voltage2 = B0005.cycle(i).data.Voltage_measured(j+1);
	    voltage3 = voltage1 - voltage2
	    if voltage3 > tvoltage && (j < 10)
		tvoltage = voltage3
	        fvoltage(counter) = tvoltage
		k(counter) = j + 1
	    end 
	end
	fcurrent(counter) =  B0005.cycle(i).data.Current_measured(k(counter));
	fresistance(counter) = fvoltage(counter) / fcurrent(counter)
    end
end 
