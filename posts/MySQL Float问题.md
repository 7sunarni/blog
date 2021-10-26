##  MySQL float问题

### 在MySQL设计数据库的时候，涉及到小数类型的时候，为了方便以后不确定出现的需求，选择decimal类型比较好。如果选择float无精度类型的话，在计算或者比较的时候就会出现精度问题。

如下测试
1. 先建立一张测试表
``` sql
CREATE TABLE `test`.`test` (
  `id` int(11) NOT NULL,
  `amount` float DEFAULT NULL,
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

2. 插入数据，然后查看一下数据
``` sql
insert into `test`.`test`(id,amount) values(1,1234567000);
insert into `test`.`test`(id,amount) values(2,2234567000);
insert into `test`.`test`(id,amount) values(3,2234567000);
select * from `test`.`test`;
```
查询的时候可以看到的是现在精度已经丢失了。

3. 假设数据库的里面的数据是上面已经丢失了的数据，现在要查询这部分数据也会有问题。
``` sql
select * from `test`.`test` where amount = 1234570000;
```
直接查询是查询不到的。
对于这种float的等值查询，用=号查询是查不到的，可以用like模糊查询。
``` sql
select * from `test`.`test` where amount like 1234570000;
```
至于为什么，是因为这个值在数据库存放的时候不是查出来显示的那个值，执行下面的sql可以看到存放的值，用decimal那个值做等值查询是可以查出来的。
``` sql
select id,amount,cast(amount as decimal(15,2)) from `test`.`test`;
```

4. 光是等值查询还好办，业务需求是找出大于等于某个amount的值，直接用>=是查出来的值可能有问题，可能会少一个。这就说明了float在比较的时候的坑了。比较好的解决办法是修改栏位的类型，将float类型改为decimal类型。

5. 但是使用的时候，选择的是有精度的float类型，在插入数据的时候会有精度损失，但是查询的时候，可以用等值查询的。所以说还是用decimal这种类型比较好，既不会出现精度损失，又方便查询。