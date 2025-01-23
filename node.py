import numpy as np
import pdb
from operators import UNARY_OPERATORS, BINARY_OPERATORS

class treeNode:
    
    def __init__(self, value, left_child = None, right_child= None):
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


    def __str__(self):

        '''
        Prints the expression
        '''

        if isinstance(self.value, (int, float, np.float64, np.int32)):
            return str(np.round(self.value,3))
        
        elif self.value in UNARY_OPERATORS:
            return f"{self.value}({self.right_child})"
        elif self.left_child is not None or self.right_child is not None:
            return f"({self.left_child} {self.value} {self.right_child})"
        else:
            return self.value
    

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
            return variables[self.value]
        
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
            

    def validate_syntax(self, variables):
        '''
        Returns a boolean which express the possibility of the current node to be a valid tree node given these conditions:
        1) A unary operator must have at least a right child
        2) A binary operator must have both childs
        3) If it is a variable it must appear in the variables array
        Params:
            variable (dictionary): lookup table for the problem variables
        Return:
            Boolean
        '''

        # check on the validity of the node value

        if not (isinstance(self.value,(int,float)) or self.value in UNARY_OPERATORS or self.value in BINARY_OPERATORS or self.value in variables):
            #print('Invalid operator')
            return False
        
        #check the operators validity

        if self.value in UNARY_OPERATORS:
            #The operator must have a right child if it is unary
            if not self.right_child:
                #print('unary op must have a right child')
                return False
            else:
                return self.right_child.validate_syntax(variables)
            
        if self.value in BINARY_OPERATORS:
            #The operator must have both childs
            if not (self.right_child!=None and self.left_child!=None):
                #print('binary operators must have both children')
                return False
            else:
                return self.left_child.validate_syntax(variables) and self.right_child.validate_syntax(variables)
        
        #the node is a leaf
        return True
    

    def validate_and_evaluate(self,lookupTable:dict):
        '''
        The expression is valid if both its syntax and the function domains are respected
        Params:
        variables (dict): The samples lookup table
        Returns:
        The evaluation (np.float64) if all is correct, False instead
        '''
        variables=list(lookupTable.keys())
        syntax_validation = self.validate_syntax(variables)
        evaluation = self.evaluate(lookupTable)
        eval_validation = not np.isnan(evaluation)

        if syntax_validation and eval_validation:
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
