# -*- coding: utf8 -*-
import pymysql


class MysqlClient(object):
    def __init__(self, host, port, user, password, db, charset="utf8"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.charset = charset
        self.connect(host, port, user, password, db, charset)

    def connect(self, host, port, user, password, db, charset):
        self.conn = pymysql.connect(host=host, port=port, user=user, password=password, database=db, charset=charset)

    def read(self, sql, args=None):
        """
        select操作使用
        :param sql: sql语句
        :param args: {tuple, None}, sql中用占位符代替的参数，按照sql中%s的顺序，为None时，需要sql中包含参数
        :return: bool, 操作是否成功，tuple，读取结果
        """
        assert isinstance(args, tuple) or isinstance(args, type(None))
        is_success = True
        results = ()
        # 用于防止连接超时导致的问题
        self.conn.ping()
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, args)
                results = cursor.fetchall()
        except Exception as e:
            is_success = False
        return is_success, results

    def write(self, sql, args=None):
        """
        写数据库，insert,update,delete操作使用
        :param sql: sql语句
        :param args: {list, tuple, None}，sql中用占位符代替的参数，按照sql中%s的顺序，sql中包含参数，args为None；需要操作多条数据时，args为list
        :return: bool，操作是否成功
        """
        is_success = True
        assert isinstance(args, list) or isinstance(args, tuple) or isinstance(args, type(None))
        # 用于防止连接超时导致的问题
        self.conn.ping()
        try:
            with self.conn.cursor() as cursor:
                if isinstance(args, list):
                    cursor.executemany(sql, args)
                else:
                    cursor.execute(sql, args)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            is_success = False
        return is_success

    def disconnect(self):
        is_success = True
        try:
            self.conn.close()
        except Exception as e:
            pass
        return is_success


if __name__ == '__main__':
    mc = MysqlClient('127.0.0.1', 3306, 'root', '123456', 'powerplant')
    sql = 'SELECT equipment, NAME, kks_code, plant, unit, speciality, system, kks_name FROM `t_account` limit 10'
    _, results = mc.read(sql)
    for result in results:
        equipment = result[0]
        name = result[1]
        kks_code = result[2]
        plant = result[3]
        print(equipment, name, kks_code, plant)