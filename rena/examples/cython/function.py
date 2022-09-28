def prime_finder_vanilla(amount):
    primes = []
    found = 0
    number = 2
    while found<amount:
        for x in primes:
            if number%x==0:
                break
        else:
            primes.append(number)

        number += 1

    return primes