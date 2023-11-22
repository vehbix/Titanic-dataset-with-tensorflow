def ortalamaBul(vektor):
    veriAdedi = len(vektor)
    if veriAdedi <= 1:
        return vektor
    else:
        return sum(vektor) / veriAdedi

def standartSapmaBul(vektor):
    sd = 0.0 # standart sapma
    veriAdedi = len(vektor)
    if veriAdedi <= 1:
        return 0.0
    else:
        for _ in vektor:
            sd += (float(_) - ortalamaBul(vektor)) ** 2
        sd = (sd / float(veriAdedi)) ** 0.5
        return sd
