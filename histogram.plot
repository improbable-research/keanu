bin(x)=0.02*floor(x/0.02)
plot 'data.out' using (bin($1)):(1.0) smooth freq with boxes
