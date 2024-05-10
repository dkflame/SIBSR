def string2list(string):
    exprlist = []
    newexplist = []
    newstring = ''
    for s in string:
        if s in ['+', '-', '*', '/', '(', ')']:
            newstring += ','+s+','
        else:
            newstring += s
    newstring = newstring.strip(',')
    newstring = newstring.replace(',,', ',')
    newstring = newstring.replace(' ', '')
    exprlist = newstring.split(',')
    for i in exprlist:
        if i.endswith('^2'):
            str_i = i.strip('^2')
            # newexplist.extend(['n2', '(', str_i, ')'])  # Use n2 to replace x^2
            newexplist.extend(['(', str_i, '*', str_i, ')'])   # Use x*x to replace x^2
        else:
            newexplist.append(i)
    return newexplist


def op2token(expression):
    pattern = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
    result = [pattern[x] if x in pattern else x for x in expression]
    return result

#判断运算符的优先级
def opOrder(op1,op2):
    order_dic = {'*':4,'$':5,'/':4,'+':3,'-':3}
    if op1 == '(' or op2 == '(':
        return False
    elif op2 == ')':
        return True
    else:
        if order_dic[op1] < order_dic[op2]:
            return False
        elif order_dic[op1] == order_dic[op2]:
            return False
        else:
            return True

#传入list
def infix2prefix(exprlist):
    prefix_list = []
    stack = []
    exprlist_reverse = []
    exprlist_len = len(exprlist)
    for i in range(exprlist_len):
        cur_token = exprlist[exprlist_len-i-1]
        if cur_token == '(':
            exprlist_reverse.append(')')
        elif cur_token == ')':
            exprlist_reverse.append('(')
        else:
            exprlist_reverse.append(cur_token)
    for s in exprlist_reverse:
        if s not in ['+', '-', '*', '/', '(', ')']:
            prefix_list.insert(0,s)
        else:
            while len(stack)  and opOrder(stack[-1],s):
                # if (opOrder(stack[-1],s) == ):  
                #     break
                # else:
                op = stack.pop()
                prefix_list.insert(0,op)
            if len(stack) == 0 or s != ')':
                stack.append(s)
            else:
                stack.pop()
        # print(prefix_list)
        # print(stack)
    while len(stack):
        prefix_list.insert(0, stack.pop())
    return prefix_list

# #输入string
# def infix2prefix(string):
#     prefix = ''
#     stack = []
#     string_tmp = ''
#     for s in string[::-1]:
#         if s == '(':
#             string_tmp += ')'
#         elif s == ')':
#             string_tmp += '('
#         else:
#             string_tmp += s
#     # print(string_tmp)
#     for s in string_tmp:
#         if s.isalpha():
#             prefix = s + prefix
#         else:
#             while len(stack)  and opOrder(stack[-1],s):
#                 op = stack.pop()
#                 prefix = op + prefix
#             if len(stack) == 0 or s != ')':
#                 stack.append(s)
#             else:
#                 stack.pop()
#     if len(stack):
#         prefix = ' '.join(stack) + prefix
#     return prefix

def expr2prefix(expr_str):
        exprlist = string2list(expr_str)
        polish_expr = infix2prefix(exprlist)
        prefix_expr = op2token(polish_expr)
        return prefix_expr

# if __name__ == '__main__':
#     # for string in ['A+B*C','(A+B)*log(C)','((A-(B+C))*D)$(E+F)']:
#     for string in ['0.0262 - 2.07*x1^2 - 1.15*x2 - 2/3/x2 + 0.0356/x3']:
#         exprlist = string2list(string)
#         polish_expr = infix2prefix(exprlist)
#         prefix_expr = op2token(polish_expr)
#         print(exprlist)
#         print(polish_expr)
#         print(prefix_expr)


