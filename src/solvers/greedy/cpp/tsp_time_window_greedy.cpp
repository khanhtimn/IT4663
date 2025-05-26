#include <algorithm>
#include <iostream>
#include <vector>


using namespace std;

struct Customer {
    int index, startTime, endTime, duration;
};

int main()
{
    int N;
    cin >> N;

    vector<Customer> customers(N);

    // Đọc thông tin khách hàng
    for (int i = 0; i < N; ++i) {
        cin >> customers[i].startTime >> customers[i].endTime >> customers[i].duration;
        customers[i].index = i + 1;
    }

    // Đọc ma trận thời gian di chuyển
    vector<vector<int>> travelTimes(N + 1, vector<int>(N + 1));
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            cin >> travelTimes[i][j];
        }
    }

    // Sắp xếp các khách hàng dựa trên thời gian bắt đầu giao hàng
    sort(customers.begin(), customers.end(), [](const Customer& a, const Customer& b) {
        return a.startTime < b.startTime;
    });

    // Lập lộ trình giao hàng
    vector<int> deliveryRoute;
    for (const auto& customer : customers) {
        deliveryRoute.push_back(customer.index);
    }

    // In kết quả
    cout << N << endl;
    for (const auto& point : deliveryRoute) {
        cout << point << " ";
    }

    return 0;
}
