fiboArray = [0,1]

def fibo(n):
    if isinstance(n, int) and n <= 0:
        print('n must be an integer and > 0.')
        return
    
    if n <= len(fiboArray):
        return fiboArray[n-1]
    else:
        temp =  fibo(n-1) + fibo(n-2)
        fiboArray.append(temp)
        return temp


print(fibo(9))
