bin(x)=0.25*floor(x/0.25)
plot 'data.out' using (bin($1)):(1.0) smooth freq with boxes
