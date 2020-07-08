#encoding=utf8
from src.token import tokens


class Parser:

    @staticmethod
    def get_tokens(info):
        if info[1] == "Year":
            return tokens.Year(info)
        if info[1] == "RelYear":
            return tokens.Year(info)
        elif info[1] == "Month":
            return tokens.Month(info)
        elif info[1] == "RelMonth":
            return tokens.Month(info)
        elif info[1] == "Day":
            return tokens.Day(info)
        elif info[1] == "Weekday":
            return tokens.Day(info)
        elif info[1] == "RelDay":
            return tokens.Day(info)
        elif info[1] == "RelWeek":
            return tokens.Weekday(info)
        elif info[1] == "HalfDay":
            return tokens.HalfDay(info)
        elif info[1] == "Hour":
            return tokens.Hour(info)
        elif info[1] == "Minute":
            return tokens.Minute(info)
        elif info[1] == "City":
            return tokens.City(info)
        elif info[1] == "Holiday":
            return tokens.Holiday(info)
        elif info[1] == "Festival":
            return tokens.Festival(info)
        elif info[1] == "Department":
            return tokens.Department(info)
        elif info[1] == "WeatherPhe":
            return tokens.WeatherPhe(info)
        else:
            return tokens.Term(info)

    def parse(self, terms):
        tokens_list = []
        for info in terms:
            t = self.get_tokens(info)
            if len(tokens_list) == 0:
                tokens_list.append(t)
            else:
                try:
                    tt = tokens_list[-1].reduce(t)
                    if tt is None:
                        tokens_list.append(t)
                    else:
                        tokens_list[-1] = tt
                except:
                    tokens_list.append(t)
        return tokens_list
