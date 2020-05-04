counter = 0
fvoltage= 0
for i = 1:length(B0005.cycle) 
    if strcmp(B0005.cycle(i).type, 'discharge')
        counter = counter+1;
	j = 1
	counter2 = 0
	ftemp = 0
	temp = 0
	for j = 1:length(B0005.cycle(i).data.Temperature_measured)
            temp = B0005.cycle(i).data.Temperature_measured(j);
	    ftemp = ftemp + temp
	    counter2 = counter2 + 1
	end
	atemp(counter) = ftemp / counter2
    end
end
