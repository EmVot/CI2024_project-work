import numpy as np
import pdb
from operators import UNARY_OPERATORS, BINARY_OPERATORS

class treeNode:
    
    def __init__(self, value, left_child = None, right_child= None, coefficient=1,):
        '''
        Initialize the tree node
        Params:
        value: The value of the node: Can be an opreand an operator or a constant
        left_child: the left subtree (default: None)
        right_child: the right subtree (default: None)
        '''
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.coefficient=coefficient


    def __str__(self):

        '''
        Prints the expression
        '''

        if isinstance(self.value, (int, float, np.float64, np.int32)):
            return str(self.value)
        
        elif self.value in UNARY_OPERATORS:
            return f"{self.value}({self.right_child})"
        elif self.left_child is not None or self.right_child is not None:
            return f"({self.left_child} {self.value} {self.right_child})"
        else:
            return str(self.coefficient)+" * "+self.value[0]+"["+self.value[1]+"]"
    

    def evaluate(self,variables):
        '''
            Recursevly evaluates the expression using numpy expressions
            Params:
                variables (dictionary): Lookup table for the variables
            Returns:
                The evaluation result (float)
        '''
        
        #pdb.set_trace()
        #Termination: the node is a leaf, which means it is a constant or a input variable
        if isinstance(self.value, (np.int64,np.float64,float,int)):
            #Leaf is a constant
            return self.value
        
        if self.value in variables:
            #Leaf is a variable -> value is represented by the variable parameter to retrieve from the lookup table
            return variables[self.value]*self.coefficient
        
        #Recursion: this node is an opertator
        
        if self.value not in UNARY_OPERATORS and self.value not in BINARY_OPERATORS:
            #Case this operator is unknown
            return ValueError(f'Unkown operator\n the value which threw the error was: {self.value}')

        if self.value in UNARY_OPERATORS:

            operand = self.right_child.evaluate(variables)

            #Case 1 : this is a unary operator
            try:
                # Domain control or critical functions
                if self.value == "arccos" and not (-1 <= operand <= 1):
                    return np.nan
                if self.value == "arcsin" and not (-1 <= operand <= 1):
                    return np.nan
                if self.value == "log" and operand <= 0:
                    return np.nan
                if self.value == "sqrt" and operand < 0:
                    return np.nan
                
                return UNARY_OPERATORS[self.value](operand)
            
            except Exception as e:
                #print(f"Error in unary function: {self.value} operating with value: {operand}: {e}")
                return np.nan
        
        else:
            left_val = self.left_child.evaluate(variables)
            right_val = self.right_child.evaluate(variables)
            try:

                if self.value=='/' and right_val == 0:
                    return np.nan
                
                elif self.value == '^':
                    
                    #avoid /0 cases
                    if left_val == 0 and right_val < 0:
                        return np.nan
                    
                    if left_val < 0 and not float(right_val).is_integer():
                        # avoid complex numbers
                        return np.nan
                    try:
                        return np.power(left_val, right_val)
                    
                    except OverflowError:
                        return np.nan  # Avoid overfolws
                
                return BINARY_OPERATORS[self.value](left_val, right_val)
            
            except Exception as e:
                #print(f"Error in binary operation {self.value} with values {left_val}, {right_val}: {e}")
                return np.nan
            

    def validate_syntax(self, variables_dict:dict):
        '''
        Returns a boolean which express the possibility of the current node to be a valid tree node given these conditions:
        1) A unary operator must have at least a right child
        2) A binary operator must have both childs
        3) If it is a variable it must appear in the variables array
        4) The expression must contain all variables
        Params:
            variable (dictionary): {
                                        key (str): variable name
                                        value (bool): it occurres in the expression
                                    }
        Return:
            Boolean
        '''

        # check on the validity of the node value

        if not (isinstance(self.value,(int,float)) or self.value in UNARY_OPERATORS or self.value in BINARY_OPERATORS or self.value in variables_dict.keys()):
            #print('Invalid operator')
            return False
        
        #check the operators validity

        if self.value in UNARY_OPERATORS:
            #The operator must have a right child if it is unary
            if not self.right_child:
                #print('unary op must have a right child')
                return False
            else:
                return self.right_child.validate_syntax(variables_dict)
            
        if self.value in BINARY_OPERATORS:
            #The operator must have both childs
            if not (self.right_child!=None and self.left_child!=None):
                #print('binary operators must have both children')
                return False
            else:
                return self.left_child.validate_syntax(variables_dict) and self.right_child.validate_syntax(variables_dict)
        
        #the node is a leaf
        if self.value in variables_dict.keys():
            #if it is a variable its occurrency value must be updated
            variables_dict[self.value]=True
        
        return True
        
    

    def validate_and_evaluate(self,lookupTable:dict):
        '''
        The expression is valid if both its syntax and the function domains are respected
        Params:
        variables (dict): The samples lookup table
        Returns:
        The evaluation (np.float64) if all is correct, False instead
        '''

        variables_occurrencies=dict(zip(lookupTable.keys(),[False]*len(lookupTable.keys())))
        syntax_validation = self.validate_syntax(variables_occurrencies)
        syntax_validation = syntax_validation and np.all(list(variables_occurrencies.values()))

        if not syntax_validation:
            return False
        
        evaluation = self.evaluate(lookupTable)
        eval_validation = not np.isnan(evaluation)

        if eval_validation:
            return evaluation
        else:
            return False
        

    def getNodes(self,nodeList:list):
        '''
            Returns the nodes as a list (array) of treeNodes
        '''

        if self is None:
            return
        
        nodeList.append(self)

        self.left_child.getNodes(nodeList)
        self.right_child.getNodes(nodeList)

    def getDepth(self,depth):
        if self.value in BINARY_OPERATORS or self.value in UNARY_OPERATORS:
            return max(self.left_child.getDepth(depth+1),self.right_child.getDepth(depth+1))
        else:
            return depth