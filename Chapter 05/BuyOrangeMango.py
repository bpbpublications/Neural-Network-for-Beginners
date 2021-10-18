from MulLayer import *

orange = 1000
orange_num = 3
mango = 1500
mango_num = 4
tax = 1.5

# layer
mul_orange_layer = MulLayer()
mul_mango_layer = MulLayer()
add_orange_mango_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
orange_price = mul_orange_layer.forward(orange, orange_num) 
mango_price = mul_mango_layer.forward(mango, mango_num)  
all_price = add_orange_mango_layer.forward(orange_price, mango_price)  
price = mul_tax_layer.forward(all_price, tax)  

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) 
dorange_price, dmango_price = add_orange_mango_layer.backward(dall_price)  
dmango, dmango_num = mul_mango_layer.backward(dmango_price)  
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  

print("price:", int(price))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dMango:", dmango)
print("dMango_num:", int(dmango_num))
print("dTax:", dtax)
