import pandas as pd
from urllib import parse
import ipaddress

features = [
    "uri_length",
    "uri_maximum_length",
    "uri_average_length",
    "parameter_avg_length",
    "parameter_value_avg_length",
    "parameter_count",
    "country_domain",
    "digit_percentage",
    "letter_percentage",
    "has_keywords",
    "is_ip",
    "has_port",
    "number_of_sub_domains",
    "host_contains_signin",
    "host_contains_blog",
    "host_contains_customerservice",
]

keywords = ["online banking", "onlinebanking", "pin", "paypal", "online", "free"]

def get_domain(url: parse.ParseResult):
    domain = url.netloc.split(":")[0]
    last_index = domain.rindex(".")
    try:
        last_domain_index = domain.rindex(".", 0, last_index)
    except ValueError:
        last_domain_index = -1
    return domain[last_domain_index+1:last_index]

def get_parameter_average_length(url: parse.ParseResult):
    param_dict = parse.parse_qs(url.params, keep_blank_values=True)
    sums = sum([sum(map(len, k)) for k, _ in param_dict])
    return sums / len(param_dict.keys)

def get_parameter_value_average_length(url: parse.ParseResult):
    param_dict = parse.parse_qs(url.params, keep_blank_values=True)
    sums = sum([sum(map(len, v)) for k, v in param_dict])
    return sums / len(param_dict.values)

def get_parameter_count(url: parse.ParseResult):
    param_dict = parse.parse_qs(url.params, keep_blank_values=True)
    return len(param_dict.keys())

def get_country_domain(url: str):
    try:
        index = url.rindex(".")
        return url[index+1:]
    except ValueError:
        return url

def has_ip(url: str):
    parsed_url = parse.urlparse(url)
    domain = parsed_url.netloc.split(":")[0]
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False

def has_port(url: str):
    parsed_url = parse.urlparse(url)
    return len(parsed_url.netloc.split(":")) > 1

def get_subdomain_count(url: str):
    parsed_url = parse.urlparse(url)
    return parsed_url.netloc.count(".") - 1

def get_url_features(df: pd.DataFrame) -> pd.DataFrame:
    df["domain"] = df["url"].apply(lambda u: get_domain(parse.urlparse(u)))
    df["country_domain"] = df["url"].apply(get_country_domain)
    df["uri_length"] = df["url"].apply(len)
    df["uri_maximum_length"] = df.groupby(by="domain").max()["uri_length"]
    df["uri_avg_length"] = df.groupby(by="domain").mean()["uri_length"]
    df["parameter_avg_length"] = df["url"].apply(get_parameter_average_length)
    df["parameter_value_avg_length"] = df["url"].apply(get_parameter_value_average_length)
    df["parameter_count"] = df["url"].apply(get_parameter_count)
    df["digit_percentage"] = df["url"].apply(lambda url: sum(1 for char in url if char.isdigit()) / len(url))
    df["letter_percentage"] = df["url"].apply(lambda url: sum(1 for char in url if char.isalpha()) / len(url))
    df["has_keywords"] = df["url"].apply(lambda url: 1 in [1 for k in keywords if k in url])
    df["is_ip"] = df["url"].apply(has_ip)
    df["has_port"] = df["url"].apply(has_port)
    df["number_of_sub_domains"] = df["url"].apply(get_subdomain_count)
    df["host_contains_signin"] = df["url"].apply(lambda url: "signin" in url)
    df["host_contains_blog"] = df["url"].apply(lambda url: "blog" in url)
    df["host_contains_customerservice"] = df["url"].apply(lambda url: "customerservic" in url)

    type_labels = pd.Categorical(
        df["type"], categories=df["type"].unique(), ordered=False).codes
    df["country_code"] = type_labels

    return df
