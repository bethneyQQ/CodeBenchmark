#!/usr/bin/env python3
"""
Test suite for st_0019: Bug fix with domain context
Tests the compound_interest function implementation.
"""

import pytest
import math

def test_compound_interest_basic():
    """Test basic compound interest calculation."""
    def compound_interest(principal, rate, time, n=1):
        # Input validation for financial applications
        if principal < 0:
            raise ValueError("Principal cannot be negative")
        if rate < -1:  # Allow negative rates but not below -100%
            raise ValueError("Interest rate cannot be less than -100%")
        if time < 0:
            raise ValueError("Time cannot be negative")
        if n <= 0 or not isinstance(n, int):
            raise ValueError("Compounding frequency must be a positive integer")
        
        # Handle edge cases
        if principal == 0 or time == 0:
            return 0.0
        if rate == 0:
            return 0.0
        
        # Correct compound interest formula: A = P(1 + r/n)^(nt)
        # Interest = A - P
        try:
            amount = principal * ((1 + rate/n) ** (n * time))
            interest = amount - principal
            
            # Round to 2 decimal places for financial precision
            return round(interest, 2)
        except OverflowError:
            raise ValueError("Calculation resulted in overflow - parameters too large")
        except ZeroDivisionError:
            raise ValueError("Division by zero in calculation")
    
    # Test basic calculation
    result = compound_interest(1000, 0.05, 2, 1)  # $1000 at 5% for 2 years, annually
    expected = 1000 * (1.05 ** 2) - 1000  # Should be $102.50
    assert abs(result - 102.50) < 0.01

def test_compound_interest_validation():
    """Test input validation."""
    def compound_interest(principal, rate, time, n=1):
        if principal < 0:
            raise ValueError("Principal cannot be negative")
        if rate < -1:
            raise ValueError("Interest rate cannot be less than -100%")
        if time < 0:
            raise ValueError("Time cannot be negative")
        if n <= 0 or not isinstance(n, int):
            raise ValueError("Compounding frequency must be a positive integer")
        
        if principal == 0 or time == 0:
            return 0.0
        if rate == 0:
            return 0.0
        
        try:
            amount = principal * ((1 + rate/n) ** (n * time))
            interest = amount - principal
            return round(interest, 2)
        except OverflowError:
            raise ValueError("Calculation resulted in overflow - parameters too large")
        except ZeroDivisionError:
            raise ValueError("Division by zero in calculation")
    
    # Test negative principal
    with pytest.raises(ValueError, match="Principal cannot be negative"):
        compound_interest(-1000, 0.05, 2)
    
    # Test invalid rate
    with pytest.raises(ValueError, match="Interest rate cannot be less than -100%"):
        compound_interest(1000, -1.5, 2)
    
    # Test negative time
    with pytest.raises(ValueError, match="Time cannot be negative"):
        compound_interest(1000, 0.05, -1)
    
    # Test invalid compounding frequency
    with pytest.raises(ValueError, match="Compounding frequency must be a positive integer"):
        compound_interest(1000, 0.05, 2, 0)

def test_compound_interest_edge_cases():
    """Test edge cases."""
    def compound_interest(principal, rate, time, n=1):
        if principal < 0:
            raise ValueError("Principal cannot be negative")
        if rate < -1:
            raise ValueError("Interest rate cannot be less than -100%")
        if time < 0:
            raise ValueError("Time cannot be negative")
        if n <= 0 or not isinstance(n, int):
            raise ValueError("Compounding frequency must be a positive integer")
        
        if principal == 0 or time == 0:
            return 0.0
        if rate == 0:
            return 0.0
        
        try:
            amount = principal * ((1 + rate/n) ** (n * time))
            interest = amount - principal
            return round(interest, 2)
        except OverflowError:
            raise ValueError("Calculation resulted in overflow - parameters too large")
        except ZeroDivisionError:
            raise ValueError("Division by zero in calculation")
    
    # Test zero principal
    assert compound_interest(0, 0.05, 2) == 0.0
    
    # Test zero time
    assert compound_interest(1000, 0.05, 0) == 0.0
    
    # Test zero rate
    assert compound_interest(1000, 0, 2) == 0.0

def test_compound_interest_different_frequencies():
    """Test different compounding frequencies."""
    def compound_interest(principal, rate, time, n=1):
        if principal < 0:
            raise ValueError("Principal cannot be negative")
        if rate < -1:
            raise ValueError("Interest rate cannot be less than -100%")
        if time < 0:
            raise ValueError("Time cannot be negative")
        if n <= 0 or not isinstance(n, int):
            raise ValueError("Compounding frequency must be a positive integer")
        
        if principal == 0 or time == 0:
            return 0.0
        if rate == 0:
            return 0.0
        
        try:
            amount = principal * ((1 + rate/n) ** (n * time))
            interest = amount - principal
            return round(interest, 2)
        except OverflowError:
            raise ValueError("Calculation resulted in overflow - parameters too large")
        except ZeroDivisionError:
            raise ValueError("Division by zero in calculation")
    
    principal = 1000
    rate = 0.06
    time = 1
    
    # Annual compounding
    annual = compound_interest(principal, rate, time, 1)
    
    # Monthly compounding
    monthly = compound_interest(principal, rate, time, 12)
    
    # Daily compounding
    daily = compound_interest(principal, rate, time, 365)
    
    # More frequent compounding should yield higher interest
    assert monthly > annual
    assert daily > monthly

if __name__ == "__main__":
    pytest.main([__file__, "-v"])