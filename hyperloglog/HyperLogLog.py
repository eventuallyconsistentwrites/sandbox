import math
import mmh3
import array

class HyperLogLog:
    def __init__(self, standardError):
        mApprox = (1.04/standardError)**2
        self.indexLength = math.floor(math.log2(mApprox))
        self.numberOfRegisters = 1 << self.indexLength
        if not self._isRegisterCountFeasible():
           raise ValueError("Number of registers has to be greater than 16")
            
        self.isSparse = True
        self.sparse = array.array('I')
        self.registers = None        
        self.sparseToDenseThreshold = self.numberOfRegisters // 4
        #print(f"m = {self.numberOfRegisters}, p = {self.indexLength}, sparseToDenseThreshold = {self.sparseToDenseThreshold}")
    
    def _isRegisterCountFeasible(self):
        return self.numberOfRegisters >= 16
        
    def _extractSparseVal(self, packed):
        bucketNumber = packed >> 8
        leftmostOnePosition = packed & 0xFF
        return bucketNumber, leftmostOnePosition
    
    def _updateSparse(self, bucketNumber, leftmostOnePosition):
        for i in range(len(self.sparse)):
            currentBucketNumber, currLeftmostOnePosition = self._extractSparseVal(self.sparse[i])
            if currentBucketNumber == bucketNumber:
                if leftmostOnePosition > currLeftmostOnePosition:
                    self.sparse[i] = (bucketNumber << 8) | leftmostOnePosition
                return
        
        # Not found: Append new packed value
        self.sparse.append((bucketNumber << 8) | leftmostOnePosition)

        # Check for conversion
        if len(self.sparse) >= self.sparseToDenseThreshold:
            self._convertToDense()

    def _convertToDense(self):
        #print("--- Converting Sparse to Dense ---")
        self.registers = bytearray(self.numberOfRegisters)
        for packed in self.sparse:
            idx, val = self._extractSparseVal(packed)
            self.registers[idx] = val
        
        self.isSparse = False
        self.sparse = None # Clear memory
    
    def _getBiasCorrectionFactor(self, numberOfRegisters):
        if numberOfRegisters == 16:
            return 0.673
        elif numberOfRegisters == 32:
            return 0.697
        elif numberOfRegisters == 64:
            return 0.709
        elif numberOfRegisters >= 128:
            return 0.7213 / (1 + (1.079 / numberOfRegisters))
        # Note: alpha value is not defined for the case where number of registers is less than 16. 
        # So here we are returning 1 which means no correction will be applied to the raw estimate.
        else:
            return 1
    
    def insertElem(self, elem):
        #print("elem: ", elem)
        hashedElem = mmh3.hash64(elem, signed=False)[0] # Binary representation of elem
        #print(" hashedElem: ", hashedElem, bin(hashedElem))
        bucketNumber = hashedElem >> (64 - self.indexLength) # Getting bucket number from first "p" bits
        #print(" bucketNumber: ", bucketNumber, bin(bucketNumber))
        remainingBits = hashedElem & ((1 << (64 - self.indexLength)) - 1) # Extract bits after first "p" bits
        #print(" remainingBits: ", remainingBits, bin(remainingBits))
        leftmostOnePosition = 64 - self.indexLength - remainingBits.bit_length() + 1 # Find (number of leading zeroes + 1) in remaining bits => This gives position of leftmost 1
        #print(" leftmostOnePosition: ", leftmostOnePosition)
        if self.isSparse:
            self._updateSparse(bucketNumber, leftmostOnePosition)
        else:
            self.registers[bucketNumber] = max(self.registers[bucketNumber], leftmostOnePosition) # Update the register
        #print(" ",self.registers)
    
    def getCardinality(self):
        
        def _getFinalEstimate(): # This returns alpha_m * m^2 * Z
            
            def _getSparseIndicator():
                indicator = 0
                numberOfEmptyRegisters = self.numberOfRegisters - len(self.sparse)
                for i in range(len(self.sparse)):
                    _, currLeftmostOnePosition = self._extractSparseVal(self.sparse[i])
                    indicator += 1.0 / (1 << currLeftmostOnePosition)
                indicator += numberOfEmptyRegisters
                indicator = 1 / indicator
                return indicator, numberOfEmptyRegisters
                
            def _getDenseIndicator():
                indicator = 0 # Z
                numberOfEmptyRegisters = 0 # V
                for i in range(self.numberOfRegisters):
                    currentRegister = self.registers[i]
                    if currentRegister == 0:
                        numberOfEmptyRegisters += 1
                    indicator += 1 / (1 << currentRegister)
                indicator = 1 / indicator
                return indicator, numberOfEmptyRegisters
            
            indicator = 0
            numberOfEmptyRegisters = 0
            if self.isSparse:
                indicator, numberOfEmptyRegisters = _getSparseIndicator()
            else:
                indicator, numberOfEmptyRegisters = _getDenseIndicator()
            harmonicMean = self.numberOfRegisters * indicator # H = m * Z
            rawEstimate = self.numberOfRegisters * harmonicMean # E_raw = m * H
            alpha = self._getBiasCorrectionFactor(self.numberOfRegisters) # alpha_m
            finalEstimate = alpha * rawEstimate # alpha_m * E_raw
            #print("E = ", finalEstimate, "V = ", numberOfEmptyRegisters)
            return finalEstimate, numberOfEmptyRegisters
        
        def _getLinearCount(numberOfEmptyRegisters):
            linearCount = self.numberOfRegisters * math.log(self.numberOfRegisters / numberOfEmptyRegisters) # E' = m * log_e(m/V)
            #print("E' = ", linearCount)
            return linearCount
        
        thresholdEstimate = (5/2) * self.numberOfRegisters
        #print("Threshold = ", thresholdEstimate)
        finalEstimate, numberOfEmptyRegisters = _getFinalEstimate()
        if finalEstimate < thresholdEstimate:
            if numberOfEmptyRegisters > 0:
                #print("Chose Linear Counting")
                finalEstimate = _getLinearCount(numberOfEmptyRegisters)
        return finalEstimate
    
    def merge(self, other):
        """
        Merges another HyperLogLog instance into this one.
        Both HLLs must have the same number of registers (m).
        """
        if self.numberOfRegisters != other.numberOfRegisters:
            raise ValueError("Cannot merge HLLs with different register counts.")
            
        # For simplicity in merging, if either is still sparse, convert to dense.
        # (In highly optimized systems, you can merge sparse-to-sparse directly)
        if self.isSparse:
            self._convertToDense()
        if other.isSparse:
            other._convertToDense()
            
        # Element-wise maximum of the registers
        for i in range(self.numberOfRegisters):
            self.registers[i] = max(self.registers[i], other.registers[i])
    
if __name__=="__main__":
    h = HyperLogLog(0.36769)
    h.insertElem("sfsefsedfs")
    h.insertElem("abfdsed")
    h.insertElem("aww")
    h.insertElem("abw")