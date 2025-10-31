def calculate_total(items, discount_percent=0):
    """Calculates the total cost of a shopping cart with an optional discount.""" 
    total = sum(item['price'] * item['quantity'] for item in items)
    if discount_percent > 0:
        # Apply the discount to find the final price.
        discount_multiplier = 1 - (discount_percent / 100)
        total *= discount_multiplier
    return total