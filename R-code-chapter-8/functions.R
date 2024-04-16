## ---------------------------
## Title: functions 
## Description: A toolbox to operate on weighted relation 
## Version: 0.1
## Original creation date: 2024-03-27
## Last modification date: 2024-04-08
## Author: Dr. Yousri El Fattah
## Email: yousri@causalcomputing.com
## ---------------------------

library(dplyr)
library(purrr)
library(rlang)



#' Elim Function
#'
#' The `elim` function eliminates a specified column from a weighted relation. 
#' The function takes a weighted relation as a data frame of two or more columns, 
#' with one distinct column named "Weight" and the name of a single column to be eliminated from the relation. 
#' The function returns the projection of the weighted relation on all columns, excluding the eliminated one.
#'
#' @param relation A data frame representing the weighted relation. 
#'   The data frame should have two or more columns with one distinct column named "Weight".
#' @param var A character string specifying the column name to be eliminated from the 'relation'.
#' @param FUN An aggregation function applied to the "Weight" of all the unique combinations of the projected variables' values. 
#'   By default, this parameter is set to 'sum'.
#' 
#' @return A data frame representing the projection of 'relation' on all columns, excluding the 'var' column.
#'
#' @examples
#' r1 <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(4,8,0,5), stringsAsFactors = FALSE)
#' elim(r1, "X")
#'
#' @importFrom stats setdiff
#' @importFrom relations project
#' @export 
elim <- function(relation, var, FUN = sum){
  # vars are all the variables in the relation scheme other than var and "Weight"
  vars <- setdiff(names(relation), c(var, "Weight"))
  
  # Return the projected relation using the vars
  return(project(relation, vars, FUN))
}

#' Project Function 
#'
#' The project function projects a weighted relation onto a subset of its attributes or variables. 
#' It aggregates the weights of the tuples that have the same attribute values in the projection.
#'
#' @param relation A data frame representing the weighted relation. The data frame should have two or more columns 
#'   with one distinct column named "Weight".
#' @param vars A character vector consisting of a subset of column names in 'relation' other than "Weight". 
#'   The function projects 'relation' onto these variables.
#' @param FUN An aggregation function applied to the "Weight" of all unique combination of 'vars' values. 
#'   By default, this parameter is set to 'sum'.
#' @return A data frame representing the projection of 'relation' with Weight set to the aggregated function 
#'   applied to the Weight of all unique combination of 'vars' values.
#'
#' @examples
#' r1 <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(4,8,0,5), stringsAsFactors = FALSE)
#' project(r1, c("X"))
#'
#' @importFrom stats setNames
#' @importFrom relations as.relation relation_projection relation_table
#' @export
project <- function(df, vars, FUN = sum){
  # Create a unique grouping key across the specified vars
  group_key <- do.call(paste, c(df[vars], sep = "_"))
  
  # Add this key to the dataframe
  df$group_key <- group_key
  
  # Split the dataframe by the key and apply the aggregation function to each group's weight
  df_grouped <- split(df, df$group_key)
  res <- do.call(rbind, lapply(df_grouped, function(group_df) {
    aggr_weight <- match.fun(FUN)(group_df$Weight)
    # Return a dataframe with the unique variable combinations and aggregated weight
    unique_rows <- unique(group_df[vars])
    unique_rows$Weight <- aggr_weight
    return(unique_rows)
  }))
  
  # Remove temporary group_key from the original dataframe
  df$group_key <- NULL
  
  return(res)
}


#' Conditional Projection Function
#'
#' The `cond` function applies a selection condition to a weighted relation 
#' and returns the resulting relation as a table. 
#' It accepts three parameters: a weighted relation (`relation`), 
#' a vector of variables (`vars`), and a vector of values (`vals`) for these variables. 
#' The `vars` and `vals` vectors must have the same length, 
#' and the `relation` should be defined on a superset of `vars`. 
#'
#' @param relation A data frame representing the weighted relation. The data frame 
#'   should have two or more columns. One of the columns should have the distinct name "Weight".
#' @param vars A character vector of variables present in `relation`.
#' @param vals A numerical vector representing value assignment for the respective variables in `vars`.
#'
#' @return A relation table obtained by selecting only tuples from `relation` that satisfy 'vars = vals'.
#'
#' @examples
#' r1 <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(4,8,0,5), stringsAsFactors = FALSE)
#' pcond(r1, c("X"), c(0))
#'
#' @importFrom relations as.relation relation_selection relation_table
#' @export

cond <- function(relation, vars, vals){
  if (length(vars) != length(vals)) {
    stop("Length of variables and values do not match.")
  }
  
  # Create the expressions
  filter_expr <- map2_chr(vars, vals, ~ paste0(.x, "==", .y)) %>%
    paste(collapse = " & ") %>%
    parse_expr()
  
  # Apply the expressions to relation
  relation <- relation %>%
    filter(!!filter_expr)
  
  return(relation)
}


#' Combine two weighted relations
#'
#' This function combines two relations by joining them while combining the weights of the joined tuples from the 
#' two relations using a combination function provided in "FUN". By default, "FUN" is multiplication.
#'
#' @param df1 A data frame representing the first weighted relation.
#' @param df2 A data frame representing the second weighted relation.
#' @param FUN A character string which is the name of the function to be combined. Defaults to "*".
#' @return A data frame representing the combined weighted relationships of the two relations.
#'
#' @examples
#' r1 <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(4,8,0,5), stringsAsFactors = FALSE)
#' r2 <- data.frame(Y = c(0,1,1), Z = c(0,0,1), Weight = c(0,5,6), stringsAsFactors = FALSE)
#' comb(r1, r2)
#'
#' @importFrom dplyr mutate select case_when
#' @export
comb <- function(df1, df2, FUN = "*"){
  common_names <- intersect(names(df1), names(df2))
  common_names <- common_names[common_names != "Weight"]
  
  join_df <- merge(x = df1, y = df2, by = common_names, all = TRUE, suffixes = c('_df1', '_df2'))
  
  if (FUN == "*") {
    # Multiply the weights
    join_df <- join_df %>% mutate(Weight = case_when(
      !is.na(Weight_df1) & !is.na(Weight_df2) ~ Weight_df1 * Weight_df2,
      !is.na(Weight_df1) & is.na(Weight_df2) ~ Weight_df1,
      is.na(Weight_df1) & !is.na(Weight_df2) ~ Weight_df2
    ))
  } else if (FUN == "+") {
    # Sum the weights
    join_df <- join_df %>% mutate(Weight = case_when(
      !is.na(Weight_df1) & !is.na(Weight_df2) ~ Weight_df1 + Weight_df2,
      !is.na(Weight_df1) & is.na(Weight_df2) ~ Weight_df1,
      is.na(Weight_df1) & !is.na(Weight_df2) ~ Weight_df2
    ))
  } else {
    stop("Invalid operation. Only + (sum) and * (multiply) are allowed.")
  }
  
  # Drop the original weight columns.
  join_df <- join_df %>% select(-Weight_df1, -Weight_df2)
  
  return(join_df)
}

#' Combine list of weighted relations
#'
#' This function takes a list of two or more relations and combine them by calling `comb`.
#'
#' @param rels A list of data frames representing weighted relations. Each data frame should have two or more columns:
#' A weigher relation should have one "Weight" column providing the weight of the relationship. 
#' Each row specifies a relationship instance.
#' @param FUN A character string which is the name of the function to be combined. Defaults to "*".
#' @return A data frame representing the combined weighted relationships of the relations list.
#'
#' @examples
#' r1 <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(4,8,0,5), stringsAsFactors = FALSE)
#' r2 <- data.frame(Y = c(0,1,1), Z = c(0,0,1), Weight = c(0,5,6), stringsAsFactors = FALSE)
#' comb_list(list(r1, r2))
#'
#' @export
comb_list <- function(rels, FUN = "*"){
  if(length(rels) == 1){
    return(rels[[1]])
  } else {
    combined_relation <- comb(rels[[1]], rels[[2]], FUN)
    
    if(length(rels) > 2){
      for(i in 3:length(rels)){
        combined_relation <- comb(combined_relation, rels[[i]], FUN)
      }
    }
    
    return(combined_relation)
  }
}

#' Variable-Relation Association Function
#'
#' The `assoc` function takes a list of weighted relations (`rels`) and a vector of variables (`vars`). 
#' It returns a dictionary whose keys are the variable names and values are those relations 
#' from `rels` where the scheme includes the variable and no variable succeeding it in the `vars` vector.
#'
#' @param rels A list of data frames each representing a weighted relation. Each data frame 
#'   should have two or more columns with one distinct column named "Weight".
#' @param vars A character vector of variable names.
#'
#' @return A list where each element key is a variable from `vars` and value is a list of relations 
#'   from `rels` where the scheme includes the variable and excludes all variables that succeed 
#'   the variable in the `vars` vector. 
#'
#' @examples
#' phi_X_Y <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(0.15, 0.6, 0.225, 0.025), stringsAsFactors = FALSE)
#' phi_Y_Z <- data.frame(Y = c(0,0,1,1), Z = c(0,1,0,1), Weight = c(1,0,0.25,0.75), stringsAsFactors = FALSE)
#' assoc(rels = list(phi_X_Y, phi_Y_Z), vars = c("X","Y","Z"))
#'
#' @export
assoc <- function(rels, vars) {
  # Initialize an empty list to store the relations
  dict <- setNames(vector("list", length(vars)), vars)
  
  # Create a named vector that maps variable names to their order
  var_order <- setNames(seq_along(vars), vars)
  
  # Loop through each variable
  for (var in vars) {
    # Loop through each relation
    for (rel in rels) {
      # Extract variables in relation directly from column names
      rel_vars <- colnames(rel)[colnames(rel) != "Weight"]
      
      # If the relation contains the variable and all variables in the relation come before or at the same position in the order
      if (var %in% rel_vars && all(var_order[rel_vars] <= var_order[var])) {
        # Add it to the dictionary
        dict[[var]] <- c(dict[[var]], list(rel))
      }
    }
  }
  
  return(dict)
}

#' Variable Removal Function
#'
#' The `remove_var` function takes a dictionary (`assoc`) that pairs variables 
#' with lists of weighted relations and a variable (`var`). The function 
#' returns a new dictionary where the entry corresponding to `var` is removed.
#'
#' @param assoc A list where each element's key is a variable and value is a 
#'   list of relations. This is typically the output of the `assoc` function.
#' @param var A character scalar representing a variable name.
#'
#' @return A new dictionary identical to `assoc` but without an entry for `var`.
#'
#' @examples
#' phi_X_Y <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), 
#'   Weight = c(0.15, 0.6, 0.225, 0.025), stringsAsFactors = FALSE)
#' phi_Y_Z <- data.frame(Y = c(0,0,1,1), Z = c(0,1,0,1), 
#'   Weight = c(1,0,0.25,0.75), stringsAsFactors = FALSE)
#' dict <- assoc(list(phi_X_Y, phi_Y_Z),c("X","Y","Z"))
#' remove_var(dict, "X")
#'
remove_var <- function(assoc, var) {
  # Create a new dictionary excluding the entry for var
  new_assoc <- assoc[!names(assoc) %in% var]
  return(new_assoc)
}


#' Relation Addition and Variable Removal Function 
#'
#' The `add_relation_remove_var` function takes a dictionary (`assoc`) that pairs variables
#' with their associated list of weighted relations, a new weighted relation (`relation`), 
#' and a variable (`var`). The function first removes the entry for `var` from `assoc`, 
#' and then updates the modified dictionary by appending the new relation to the list of relations belonging
#' to the highest ordered variable present in the new relation.
#'
#' @param assoc A list where each element's key is a variable and value is a 
#'   list of relations. This is typically the output of the `assoc` function.
#' @param relation A data frame representing a new weighted relation to be added. 
#'   The data frame should contain columns for variable values and "Weight".
#' @param var A character scalar representing the variable name to be removed from `assoc`.
#'
#' @return A new dictionary identical to `assoc` but without an entry for `var` and with `relation` added to 
#'   the list of relations of the highest ordered variable in `relation`.
#'
#' @examples
#' f_X <- data.frame(X = c(0,0,1), Weight = c(3,5,7), stringsAsFactors = FALSE)
#' f_Y <- data.frame(Y = c(1,0,1), Weight = c(2,10,6), stringsAsFactors = FALSE)
#' f_Z <- data.frame(Z = c(1,1,0), Weight = c(4,3,1), stringsAsFactors = FALSE)
#' dict <- assoc(list(f_X,f_Y,f_Z),c("X","Y","Z"))
#' f_X_Y <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(9,3,4,7), stringsAsFactors = FALSE)
#' dict2 <- add_relation1(dict, f_X_Y, "Z")
#' 
add_relation_remove_var <- function(assoc, relation, var) {
  # remove var from assoc dictionary
  assoc1 <- remove_var(assoc, var) 
  # add relation to assoc1 dictionary
  assoc2 <- add_relation(assoc1, relation)
  return (assoc2)
}

#' Relation Addition Function
#'
#' The `add_relation` function takes a dictionary (`assoc`) that pairs variables
#' with their associated list of weighted relations and a new weighted relation (`relation`). 
#' The function updates the dictionary by appending the new relation to the list of relations 
#' belonging to the highest ordered variable present in the new relation.
#'
#' @param assoc A list where each element's key is a variable and value is a 
#'   list of relations. This is typically the output of the `assoc` function.
#' @param relation A data frame representing a new weighted relation to be added. 
#'   The data frame should contain columns for variable values and "Weight".
#'
#' @return A new dictionary identical to `assoc` but with `relation` added to 
#'   the list of relations of the highest ordered variable in `relation`.
#'
#' @examples
#' f_X <- data.frame(X = c(0,0,1), Weight = c(3,5,7), stringsAsFactors = FALSE)
#' f_Y <- data.frame(Y = c(1,0,1), Weight = c(2,10,6), stringsAsFactors = FALSE)
#' f_Z <- data.frame(Z = c(1,1,0), Weight = c(4,3,1), stringsAsFactors = FALSE)
#' dict <- assoc(list(f_X,f_Y,f_Z),c("X","Y","Z"))
#' f_X_Y <- data.frame(X = c(0,0,1,1), Y = c(0,1,0,1), Weight = c(9,3,4,7), stringsAsFactors = FALSE)
#' dict2 <- add_relation(dict, f_X_Y)
#'
add_relation <- function(assoc, relation) {
  # Extract variables from the relation
  rel_vars <- colnames(relation)[colnames(relation) != "Weight"]
  
  # Create a map from variable name to their order
  var_order <- setNames(seq_along(names(assoc)), names(assoc))
  
  # Identify the variable in the relation with the highest order
  highest_var <- rel_vars[which.max(var_order[rel_vars])]
  
  # Add relation only to the variable with the highest order
  if (highest_var %in% names(assoc)) {
    assoc[[highest_var]] <- c(assoc[[highest_var]], list(relation))
  }
  
  return(assoc)
}


