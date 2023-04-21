function [R2] = Rsquared(predicted,actual,NameValueArgs)
%Calculate R^2, also called the Coefficient of Determination. 
%   R2 = 1 - (SSResid/SSTot)
%   Can be applied along a single dimension or all dimensions.

arguments
    predicted           double
    actual              double {mustBeEqualSize(predicted,actual)}
    NameValueArgs.dim   (1,1) {mustBeValidDim(NameValueArgs.dim,predicted)} = "all"
end

dim = NameValueArgs.dim;

sumSqrResiduals = sum((actual - predicted).^2,dim);
sumSqrTotal = sum((actual - mean(actual,dim)).^2,dim);
R2 = 1 - (sumSqrResiduals./sumSqrTotal);
end



function mustBeEqualSize(a,b)
    if size(a)~=size(b)
        eid = 'Size:notEqual';
        msg = 'Size of first input must equal size of second input.';
        throwAsCaller(MException(eid,msg))
    end
end

function mustBeValidDim(dim,data)
    if isstring(dim) 
        if ~(strcmp(dim,"all"))
            eid = 'Dimension:notValid';
            msg = 'Must provide a valid dimension (positive integer or "all").';
            throwAsCaller(MException(eid,msg))
        end
    elseif isnumeric(dim)
        if ~((dim==floor(dim)) && (dim > 0))
            eid = 'Dimension:notValid';
            msg = 'Must provide a valid dimension (positive integer or "all").';
            throwAsCaller(MException(eid,msg))
        elseif ~(dim <= ndims(data))
            eid = 'Dimension:notValid';
            msg = 'Dimension exceeds size of data.';
            throwAsCaller(MException(eid,msg))
        end
    else
        eid = 'Dimension:notValid';
        msg = 'Must provide a valid dimension (positive integer or "all").';
        throwAsCaller(MException(eid,msg))
    end
end