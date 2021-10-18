from MulLayer import *


orange = 1000
orange_num = 3
tax = 1.5

mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
orange_price = mul_orange_layer.forward(orange, orange_num)
price = mul_tax_layer.forward(orange_price, tax)

# backward
dprice = 1
dorange_price, dtax = mul_tax_layer.backward(dprice)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print("price:", int(price))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)
